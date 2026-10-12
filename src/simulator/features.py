"""単勝モデル戦略（締切直前の資金流入 × 高オッズ）の特徴量を構築するモジュール。

検証内容は ``docs/report/anaba_strategy_report/anaba_strategy_report.md``。
戦略の核心は1行で書ける::

    締切直前に、10〜150倍で、直前の数分間にオッズが下がった馬を買う。

本モジュールは学習側（:mod:`src.simulator.retrain`）と推論側
（:mod:`src.simulator.predict_today`）の双方から呼ばれる。特徴量の定義を
1箇所に閉じ込めることで、学習時と推論時のずれを構造的に防ぐ。

IMPORTANT (学習は 1m / 当日判断は 2m という時点のずれ):
    利用者の指示により、当日の判断時点は **2m**（発走2分前）である。
    一方で過去データ ``data/processed/odds_series/*_tansho.csv`` は
    60m/30m/10m/5m/3m/1m の6時点しか持たず、しかも実測すると
    **``3m`` 列は全月・全行で ``5m`` と完全に同一**であった
    （JV-Link の履歴が実質 5m → 1m しか返していない）。
    よって過去データから 2m スナップショットを作る手段が無い。

    そのため本モジュールは「締切直前時点」を抽象化し、
    学習では ``1m``、当日推論では ``2m`` を同じ役割の列として扱う。
    どちらも「5m から締切直前までにオッズがどう動いたか」を測る点で
    意味は同じだが、**区間の長さが4分と3分で異なる**ことは既知のずれである。
    ドリフトは区間長に比例して大きくなるため、2m のドリフトは 1m より
    わずかに小さく出る（＝採用がやや保守的になる）側のずれである。

IMPORTANT (時系列リーク防止):
    ここで参照するのは締切前の市場スナップショットと、発走前に確定している
    レース属性のみである。着順・確定オッズ・確定人気・払戻は一切参照しない。

NOTE (29列ではなく22列である理由):
    レポートの検証は29列で行われたが、このリポジトリでは7列が使えない。

    ======================================  ===================================
    除外した列                              理由
    ======================================  ===================================
    ``distance`` ``is_turf`` ``is_dirt``    データが存在しない
                                            （``data/processed/races`` が無く、
                                            ``horses`` CSV にも距離・馬場が無い）
    ``age`` ``impost``                      **当日取得できない**。学習側の
    ``horse_weight``                        ``horses`` CSV にはあるが、当日は
    ``horse_weight_change``                 出馬表 CSV に無く、出馬表パーサも
                                            このリポジトリに無い。推論時に必ず
                                            NaN になる列を学習に入れると、
                                            学習時と当日で入力分布が黙ってずれる
    ======================================  ===================================

    よって**市場スナップショットのみの22列**で実装する。これは意図的な選択で
    ある。レポートの特徴量削減の節によれば gain importance の 89.9% は上位4列
    （``inc_share`` / ``log_odds`` / ``mkt`` / ``mkt_ratio_to_fav``）が占め、
    下位15列は合計2%未満で、8列に削っても回収率139.9%・AUC 0.8432 と29列版
    （139.8% / 0.8431）に差が無かった。除外した7列はいずれもその下位側にある。

    NOTE: ``horse_weight`` はレポートの8列版に含まれるが、当日取得できない
    ため本実装では使えない。出馬表から馬体重を取得できるようになれば、
    :data:`FEATURE_COLS` に戻して再学習する価値がある。
"""
import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

# 票数の逆算・レース内集約は学習データのローダと同じ関数を使う
# （学習時と推論時で投票額の定義がずれないようにするため）。
from src.simulator.odds_series_loader import (
    EPS,
    TAKEOUT_RATE,
    estimate_votes,
    race_broadcast,
    safe_share,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# スナップショットの役割
# ---------------------------------------------------------------------------

# 学習で「締切直前」として使う時点（過去データに 2m が無いため 1m）
TRAIN_DECISION_SNAPSHOT: str = '1m'

# 当日の判断で使う時点（利用者の指示により 2m）
LIVE_DECISION_SNAPSHOT: str = '2m'

# ドリフト・流入シェアの起点。締切直前と対になる時点。
DRIFT_BASE_SNAPSHOT: str = '5m'

# 特徴量の構築に必要な全スナップショット（締切直前時点は別に足す）
CONTEXT_SNAPSHOTS: Tuple[str, ...] = ('60m', '30m', '10m', '5m')

# 特徴量の内部での締切直前時点の呼び名。学習/推論のどちらでも
# この名前の列に正規化してから特徴量を作る（列名に 1m/2m を残さない）。
LATE: str = 'late'

# ---------------------------------------------------------------------------
# 列名
# ---------------------------------------------------------------------------

# 判断に使う単勝オッズ（EV 計算のオッズでもある）
ODDS_COL: str = f'odds_{LATE}'

# 5m → 締切直前のログオッズ変化。負なら「オッズが下がった＝資金が入った」
DRIFT_COL: str = 'drift_5_late'

# レース選別に使う締切直前時点のレース総投票額（円）
POOL_COL: str = f'pool_{LATE}_yen'

# 行を一意に決めるキー
KEY_COLS: Tuple[str, str] = ('race_id', 'horse_number')

# 票数合計（hyosu_total）は百円単位なので、円に直す倍率
POOL_YEN_PER_UNIT: float = 100.0

# ---------------------------------------------------------------------------
# モデル入力（22列）
# ---------------------------------------------------------------------------

# レポートの29列から、取得できない7列を除いた22列（上の NOTE を参照）。
# 列を増減したら再学習が必要（artifacts.load_params が不一致を検知する）。
FEATURE_COLS: Tuple[str, ...] = (
    # --- 水準（締切直前のオッズと市場確率） ---
    'log_odds_late',
    'log_odds_5m',
    'mkt_late',
    'mkt_rank_late',
    # --- ドリフト（この戦略の中核。各区間のログオッズ変化） ---
    DRIFT_COL,
    'drift_10_5',
    'drift_30_10',
    'drift_60_30',
    'drift_60_late',
    'accel',
    'drift_5_late_rel',
    'drift_10_5_rel',
    # --- 直前区間の新規資金シェア（単独で gain の65%を占める列） ---
    'inc_share_5_late',
    'inc_vs_mkt',
    'inc_ratio',
    # --- プール規模 ---
    'log_pool_late',
    'pool_growth_5_late',
    'log_pool_per_horse',
    # --- レース内の分散 ---
    'hhi_late',
    'fav_strength',
    'mkt_ratio_to_fav',
    # --- レース属性（発走前に確定しており、当日も出走表から分かる） ---
    'field_size',
)


def required_snapshots(decision_snapshot: str) -> Tuple[str, ...]:
    """特徴量の構築に必要なスナップショット名を返す。

    Args:
        decision_snapshot: 締切直前として使う時点（``1m`` または ``2m``）

    Returns:
        Tuple[str, ...]: 読み込むべきスナップショット（時系列順）
    """
    snaps = list(CONTEXT_SNAPSHOTS)
    if decision_snapshot not in snaps:
        snaps.append(decision_snapshot)
    return tuple(snaps)


def _race_transform(df: pd.DataFrame, col: str, how: str) -> pd.Series:
    """レース内集約を行う（``groupby.transform`` の薄いラッパ）。"""
    return df.groupby('race_id')[col].transform(how)


def normalize_late_columns(
    df: pd.DataFrame, decision_snapshot: str
) -> pd.DataFrame:
    """``odds_{decision}_raw`` を ``odds_late_raw`` に正規化する。

    学習（1m）と推論（2m）で同じ特徴量コードを通すための橋渡しである。
    元の列は残すので、CSV 出力で実際の時点を確認できる。

    Args:
        df: ``odds_{decision}_raw`` / ``pool_{decision}_raw`` を持つ DataFrame
        decision_snapshot: 締切直前として使う時点（``1m`` または ``2m``）

    Returns:
        pd.DataFrame: ``odds_late_raw`` / ``pool_late_raw`` を加えた DataFrame

    Raises:
        ValueError: 指定時点の列が無い場合
    """
    out = df.copy()
    for kind in ('odds', 'pool'):
        src = f'{kind}_{decision_snapshot}_raw'
        if src not in out.columns:
            raise ValueError(
                f'締切直前時点の列がありません: {src}。'
                f' {decision_snapshot} のスナップショットを読み込んでいるか'
                ' 確認してください。'
            )
        out[f'{kind}_{LATE}_raw'] = out[src].astype('float64')
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """市場スナップショットから22列の特徴量を作る。

    投票額は JRA の単勝オッズの定義式から逆算する::

        odds_i = (1 - 0.20) * レース総投票額 / 投票額_i
        → 投票額_i = 0.8 * レース総投票額 / odds_i

    Args:
        df: ``race_id`` / ``horse_number`` と、``odds_{snap}_raw`` /
            ``pool_{snap}_raw``（``60m/30m/10m/5m/late``）を持つ DataFrame

    Returns:
        pd.DataFrame: ``FEATURE_COLS`` と ``ODDS_COL`` / ``DRIFT_COL`` /
            ``POOL_COL`` を加えた DataFrame（入力は破壊しない。行順は保つ）

    Raises:
        ValueError: 必要な列が欠けている場合
    """
    snaps: Tuple[str, ...] = CONTEXT_SNAPSHOTS + (LATE,)
    need = list(KEY_COLS) + [
        f'{kind}_{snap}_raw' for snap in snaps for kind in ('odds', 'pool')
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f'特徴量の構築に必要な列がありません: {missing}。'
            ' スナップショットを読み込んでいるか確認してください。'
        )

    d = df.copy().reset_index(drop=True)

    # --- 時点別の推定投票額と市場確率シェア ---
    for snap in snaps:
        odds = d[f'odds_{snap}_raw'].to_numpy(dtype='float64')
        pool = d[f'pool_{snap}_raw'].to_numpy(dtype='float64')
        d[f'votes_{snap}'] = estimate_votes(odds, pool)
        # 市場確率は 1/オッズ をレース内で再正規化したもの（控除率を吸収する）
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(odds > EPS, 1.0 / np.where(odds > EPS, odds, 1.0), np.nan)
        total = race_broadcast(d, np.nan_to_num(inv, nan=0.0))
        d[f'mkt_{snap}'] = safe_share(inv, total)

    # --- 各区間のログオッズ変化（この戦略の中核シグナル） ---
    def _drift(after: str, before: str) -> np.ndarray:
        a = d[f'odds_{after}_raw'].to_numpy(dtype='float64')
        b = d[f'odds_{before}_raw'].to_numpy(dtype='float64')
        valid = (a > EPS) & (b > EPS)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(valid, np.log(np.where(valid, a / b, 1.0)), np.nan)

    d[DRIFT_COL] = _drift(LATE, DRIFT_BASE_SNAPSHOT)
    d['drift_10_5'] = _drift('5m', '10m')
    d['drift_30_10'] = _drift('10m', '30m')
    d['drift_60_30'] = _drift('30m', '60m')
    d['drift_60_late'] = _drift(LATE, '60m')
    # 加速度: 直前の動きが新しいものか、前からの継続か
    d['accel'] = d[DRIFT_COL] - d['drift_10_5']

    # --- レース全体のドリフトに対する相対値（レースごと丸ごと動くことがある） ---
    for col in (DRIFT_COL, 'drift_10_5'):
        d[f'{col}_rel'] = d[col] - _race_transform(d, col, 'median')

    # --- 直前区間に入った新規資金のうち、この馬が占めたシェア ---
    inc = (d[f'votes_{LATE}'] - d[f'votes_{DRIFT_BASE_SNAPSHOT}']).to_numpy(
        dtype='float64'
    )
    # 累積量なので増分は非負が期待値。負値（丸め・欠測由来）は 0 に潰す。
    inc = np.where(np.isfinite(inc), np.maximum(inc, 0.0), np.nan)
    inc_total = race_broadcast(d, np.nan_to_num(inc, nan=0.0))
    d['inc_share_5_late'] = safe_share(inc, inc_total)
    # 既存の資金シェアより大きいシェアの新規資金を集めているか
    base_mkt = d[f'mkt_{DRIFT_BASE_SNAPSHOT}']
    d['inc_vs_mkt'] = d['inc_share_5_late'] - base_mkt
    d['inc_ratio'] = d['inc_share_5_late'] / base_mkt.where(base_mkt > EPS)

    # --- 水準 ---
    late_odds = d[f'odds_{LATE}_raw'].astype('float64')
    base_odds = d[f'odds_{DRIFT_BASE_SNAPSHOT}_raw'].astype('float64')
    d[ODDS_COL] = late_odds
    with np.errstate(divide='ignore', invalid='ignore'):
        d['log_odds_late'] = np.log(late_odds.where(late_odds > EPS))
        d['log_odds_5m'] = np.log(base_odds.where(base_odds > EPS))
    d['mkt_late'] = d[f'mkt_{LATE}']
    # 市場人気順（オッズの小さい順）。同値は最小順位に揃える。
    d['mkt_rank_late'] = d.groupby('race_id')[f'odds_{LATE}_raw'].rank(method='min')

    # --- プール規模（百円単位 → 円） ---
    late_pool = d[f'pool_{LATE}_raw'].astype('float64') * POOL_YEN_PER_UNIT
    d[POOL_COL] = late_pool.where(late_pool > EPS)
    d['field_size'] = _race_transform(d, 'horse_number', 'size').astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        d['log_pool_late'] = np.log(d[POOL_COL])
        d['log_pool_per_horse'] = np.log(d[POOL_COL] / d['field_size'])
    base_pool = d[f'pool_{DRIFT_BASE_SNAPSHOT}_raw'].astype('float64')
    d['pool_growth_5_late'] = (
        d[f'pool_{LATE}_raw'].astype('float64') / base_pool.where(base_pool > EPS)
    )

    # --- レース内の分散 ---
    d['hhi_late'] = race_broadcast(
        d, (d[f'mkt_{LATE}'] ** 2).to_numpy(dtype='float64'), 'sum'
    )
    fav = _race_transform(d, f'mkt_{LATE}', 'max')
    d['fav_strength'] = fav
    d['mkt_ratio_to_fav'] = d[f'mkt_{LATE}'] / fav.where(fav > EPS)

    return d


def drop_incomplete_rows(
    df: pd.DataFrame, *, context: str, required: Sequence[str] | None = None
) -> pd.DataFrame:
    """特徴量が算出できない行を、件数をログに残したうえで除外する。

    欠損を 0 等で補完すると「動きが無かった」と誤って断定することになり、
    ``inc_share_5_late`` や ``drift_5_late`` の意味が壊れる。
    よって補完せず除外する。

    Args:
        df: 特徴量を持つ DataFrame
        context: ログに出す文脈（'学習' / '当日予測' 等）
        required: 必須列（省略時は賭け判断に不可欠な列）

    Returns:
        pd.DataFrame: 必須列が全て有効な行のみ（index は振り直す）

    Raises:
        ValueError: 有効な行が1行も残らない場合
    """
    if required is None:
        # 22列はすべて市場スナップショットから作る列なので、全てを必須にする
        required = list(dict.fromkeys(
            list(FEATURE_COLS) + [ODDS_COL, DRIFT_COL, POOL_COL]
        ))

    cols = [c for c in required if c in df.columns]
    valid = df[cols].notna().all(axis=1)
    n_dropped = int((~valid).sum())
    reasons: Dict[str, int] = {
        col: int(df[col].isna().sum())
        for col in cols if int(df[col].isna().sum())
    }
    if n_dropped:
        logger.warning(
            '[%s] 特徴量が算出できない %s 行を除外します（内訳: %s）',
            context, f'{n_dropped:,}', reasons,
        )
    out = df[valid].reset_index(drop=True)
    if not len(out):
        # reasons はこの関数内で欠損が発覚した列の内訳。全行NaNの列（＝
        # そのスナップショット自体が1件も取得できていない）を具体的に示す。
        missing_cols = sorted(
            col for col, count in reasons.items() if count == len(df)
        )
        detail = (
            f'（全行欠損の列: {", ".join(missing_cols)}）' if missing_cols
            else f'（内訳: {reasons}）'
        )
        raise ValueError(
            f'[{context}] 特徴量が算出できる行が1行もありません。{detail}'
            ' 必要な各時点のオッズ・総票数スナップショットが'
            '（特に発走直前に起動した場合は早い時点ほど）取得できているか'
            '確認してください。'
        )
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """モデルへ渡す特徴量だけを列順を固定して取り出す。

    学習時と推論時で列順が食い違うと LightGBM の予測が黙って壊れるため、
    双方をこの関数に通す。

    Args:
        df: ``FEATURE_COLS`` を含む DataFrame

    Returns:
        pd.DataFrame: ``FEATURE_COLS`` の順に並べた特徴量

    Raises:
        ValueError: 特徴量カラムが欠けている場合
    """
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'特徴量カラムがありません: {missing}')
    return df[list(FEATURE_COLS)]


def race_pool(df: pd.DataFrame) -> pd.Series:
    """レース単位の締切直前プール（円）を取り出す（同一レース内は同値）。

    Args:
        df: ``race_id`` と ``POOL_COL`` を持つ DataFrame

    Returns:
        pd.Series: race_id を index とするプール（円）
    """
    return df.groupby('race_id')[POOL_COL].first()


def describe_pool_quantiles(pool_by_race: pd.Series) -> List[Tuple[str, float]]:
    """レース総投票額の分位点一覧を作る（しきい値の位置を確認する補助）。

    Args:
        pool_by_race: レース単位のプール（円）

    Returns:
        List[Tuple[str, float]]: (分位点ラベル, 金額) の一覧
    """
    values = pd.to_numeric(pool_by_race, errors='coerce').dropna()
    if not len(values):
        raise ValueError(f'{POOL_COL} が有効なレースが1件もありません')
    return [
        (f'q{q:.2f}', float(values.quantile(q)))
        for q in (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)
    ]


def takeout_rate() -> float:
    """票数逆算に使っている JRA 控除率を返す（ログ・検証用）。

    Returns:
        float: 控除率（単勝 0.20）
    """
    return TAKEOUT_RATE
