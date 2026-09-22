"""当日予測・再学習で共用する市場特徴量の構築モジュール。

``docs/pool_filter_condition_spec.md`` §3 の2列だけを作る。

    ``odds_1m``            1m時点の単勝オッズ（生の値。変換しない）＝市場の水準
    ``inc_share_5m_1m``    直前1分の流入額がレース内で占めるシェア＝資金の勢い

加えてレース選別に使う ``pool_1m``（1m時点のレース総投票額・円）を作る。
``pool_1m`` は**特徴量ではなくフィルタ**である。レース内の全馬に同じ値が
入るため馬の序列化に寄与せず、モデルに入れると回収率が -22〜-31ポイント
悪化することが実測されている（spec §5.4）。列名も ``FEATURE_COLS`` から
外し、誤ってモデル入力に混ざらないようにしてある。

本モジュールは学習側（:mod:`src.simulator.retrain`）と推論側
（:mod:`src.simulator.predict_today`）の双方から呼ばれる。特徴量の定義を
1箇所に閉じ込めることで、学習時と推論時のずれを構造的に防ぐ。

IMPORTANT (時系列リーク防止):
    ここで参照するのは 5m / 1m 時点の締切前スナップショットのみである。
    確定オッズ・確定票数・着順・払戻は一切参照しない。
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# 以前は ``src.features.market.odds_series_loader`` の補助関数を参照していたが、
# 現在の配布ツリーには同パッケージが含まれない。予測・学習で同じ定義を使う
# という性質を保ったまま、この小さな数式を本モジュールへ集約する。
EPS: float = 1e-12
TAKEOUT_RATE: float = 0.20


def estimate_votes(odds: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """単勝オッズと総票数から馬別票数を逆算する。"""
    odds_array = np.asarray(odds, dtype="float64")
    pool_array = np.asarray(pool, dtype="float64")
    valid = (
        np.isfinite(odds_array)
        & np.isfinite(pool_array)
        & (odds_array > EPS)
        & (pool_array >= 0)
    )
    out = np.full(odds_array.shape, np.nan, dtype="float64")
    out[valid] = (1.0 - TAKEOUT_RATE) * pool_array[valid] / odds_array[valid]
    return out


def race_broadcast(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    """馬別値のレース合計を各馬行へ戻す。"""
    work = pd.DataFrame(
        {"race_id": frame["race_id"].to_numpy(), "value": np.asarray(values)}
    )
    return work.groupby("race_id", sort=False)["value"].transform("sum").to_numpy()


def safe_share(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """有効な正の分母に対してのみ比率を返す。"""
    top = np.asarray(numerator, dtype="float64")
    bottom = np.asarray(denominator, dtype="float64")
    valid = np.isfinite(top) & np.isfinite(bottom) & (bottom > EPS)
    out = np.full(top.shape, np.nan, dtype="float64")
    out[valid] = top[valid] / bottom[valid]
    return out

logger = logging.getLogger(__name__)

# モデル入力に使う特徴量（spec §3.2）。この2列以外を足してはならない。
FEATURE_COLS: Tuple[str, ...] = ('odds_1m', 'inc_share_5m_1m')

# レース選別に使う列。特徴量ではない（spec §5.4）。
POOL_COL: str = 'pool_1m'

# 特徴量の構築に必要なスナップショット
REQUIRED_SNAPSHOTS: Tuple[str, str] = ('5m', '1m')

# 行を一意に決めるキー
KEY_COLS: Tuple[str, str] = ('race_id', 'horse_number')

# 票数合計（hyosu_total）は百円単位なので、円に直す倍率
POOL_YEN_PER_UNIT: float = 100.0


def build_features(
    odds_5m: np.ndarray,
    pool_5m: np.ndarray,
    odds_1m: np.ndarray,
    pool_1m: np.ndarray,
    race_id: pd.Series,
) -> pd.DataFrame:
    """5m / 1m のオッズ・総票数から spec §3 の2特徴量と pool_1m を作る。

    投票額は JRA の単勝オッズの定義式から逆算する（spec §3.1）::

        odds_i = (1 - 0.20) * レース総投票額 / 投票額_i
        → 投票額_i = 0.8 * レース総投票額 / odds_i

    ``inc_share_5m_1m`` は「5m→1m の間にレース全体へ入った新規資金のうち、
    この馬が占めた割合」である::

        inc_i  = max(投票額_1m,i - 投票額_5m,i, 0)
        share  = inc_i / Σ_j inc_j

    Args:
        odds_5m: 5m時点の単勝オッズ（非正・欠損は NaN であること）
        pool_5m: 5m時点のレース総票数（百円単位）
        odds_1m: 1m時点の単勝オッズ
        pool_1m: 1m時点のレース総票数（百円単位）
        race_id: 各行の race_id（レース内集約に使う。行順は他引数と揃えること）

    Returns:
        pd.DataFrame: ``FEATURE_COLS`` + ``POOL_COL`` を持つ DataFrame
            （行順は入力と同じ。算出不能な行は NaN）

    Raises:
        ValueError: 配列の長さが揃っていない場合
    """
    lengths = {
        'odds_5m': len(odds_5m), 'pool_5m': len(pool_5m),
        'odds_1m': len(odds_1m), 'pool_1m': len(pool_1m),
        'race_id': len(race_id),
    }
    if len(set(lengths.values())) != 1:
        raise ValueError(f'入力配列の長さが揃っていません: {lengths}')

    # 馬番別の推定投票額（百円単位）。オッズ・総票数が無効な行は NaN のまま。
    votes_5m = estimate_votes(odds_5m, pool_5m)
    votes_1m = estimate_votes(odds_1m, pool_1m)

    # 累積量なので増分は非負が期待値。負値（丸め・欠測由来）は 0 に潰す。
    # spec §3.2 の max(投票額_1m - 投票額_5m, 0) と同じ扱い。
    inc = votes_1m - votes_5m
    inc = np.where(np.isfinite(inc), np.maximum(inc, 0.0), np.nan)

    frame = pd.DataFrame({'race_id': np.asarray(race_id)})
    inc_total = race_broadcast(frame, inc)

    out = pd.DataFrame({
        'odds_1m': np.asarray(odds_1m, dtype='float64'),
        'inc_share_5m_1m': safe_share(inc, inc_total),
        # 百円単位の票数合計を円に直す（spec §5.1）
        POOL_COL: np.asarray(pool_1m, dtype='float64') * POOL_YEN_PER_UNIT,
    })
    out[POOL_COL] = out[POOL_COL].where(out[POOL_COL] > EPS)
    return out


def attach_features(df: pd.DataFrame) -> pd.DataFrame:
    """``odds_{5m,1m}_raw`` / ``pool_{5m,1m}_raw`` を持つ表に特徴量を付ける。

    :func:`src.features.market.odds_series_loader.load_odds_series` が返す
    列名（``odds_5m_raw`` 等）をそのまま受け取る。

    Args:
        df: ``race_id`` / ``horse_number`` と 5m・1m の生列を持つ DataFrame

    Returns:
        pd.DataFrame: ``FEATURE_COLS`` + ``POOL_COL`` を加えた DataFrame
            （入力は破壊しない）

    Raises:
        ValueError: 必要な列が欠けている場合
    """
    need = list(KEY_COLS) + [
        f'{kind}_{snap}_raw'
        for snap in REQUIRED_SNAPSHOTS for kind in ('odds', 'pool')
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f'特徴量の構築に必要な列がありません: {missing}。'
            ' 5m と 1m のスナップショットを読み込んでいるか確認してください。'
        )

    out = df.copy().reset_index(drop=True)
    built = build_features(
        out['odds_5m_raw'].to_numpy(dtype='float64'),
        out['pool_5m_raw'].to_numpy(dtype='float64'),
        out['odds_1m_raw'].to_numpy(dtype='float64'),
        out['pool_1m_raw'].to_numpy(dtype='float64'),
        out['race_id'],
    )
    for col in built.columns:
        out[col] = built[col].to_numpy()
    return out


def drop_incomplete_rows(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """特徴量が算出できない行を、件数をログに残したうえで除外する。

    欠損を 0 等で補完すると「動きが無かった」と誤って断定することになり、
    ``inc_share_5m_1m`` の意味が壊れる。よって補完せず除外する。

    Args:
        df: ``FEATURE_COLS`` と ``POOL_COL`` を持つ DataFrame
        context: ログに出す文脈（'学習' / '当日予測' 等）

    Returns:
        pd.DataFrame: 全特徴量と pool_1m が有効な行のみ（index は振り直す）

    Raises:
        ValueError: 有効な行が1行も残らない場合
    """
    cols = list(FEATURE_COLS) + [POOL_COL]
    valid = df[cols].notna().all(axis=1)
    n_dropped = int((~valid).sum())
    if n_dropped:
        reasons: Dict[str, int] = {
            col: int(df[col].isna().sum()) for col in cols
        }
        logger.warning(
            '[%s] 特徴量が算出できない %s 行を除外します（内訳: %s）',
            context, f'{n_dropped:,}', reasons,
        )
    out = df[valid].reset_index(drop=True)
    if not len(out):
        raise ValueError(
            f'[{context}] 特徴量が算出できる行が1行もありません。'
            ' 5m / 1m のオッズと総票数が取得できているか確認してください。'
        )
    return out


def pool_threshold(pool_by_race: pd.Series, quantile: float) -> float:
    """レース総投票額の分位点からフィルタのしきい値を求める（spec §5.2）。

    **レース単位**の分布から取る。馬単位のまま分位点を取ると出走頭数の
    多いレースが重く数えられ、しきい値がずれる。

    Args:
        pool_by_race: レース単位の ``pool_1m``（円）
        quantile: 分位点（spec の採用値は 0.75）

    Returns:
        float: しきい値（円）

    Raises:
        ValueError: 分位点が [0, 1] の範囲外、または有効なレースが無い場合
    """
    if not 0.0 <= quantile <= 1.0:
        raise ValueError(f'quantile は 0〜1 で指定してください: {quantile}')
    values = pd.to_numeric(pool_by_race, errors='coerce').dropna()
    if not len(values):
        raise ValueError('pool_1m が有効なレースが1件もありません')
    return float(values.quantile(quantile))


def race_pool(df: pd.DataFrame) -> pd.Series:
    """レース単位の ``pool_1m`` を取り出す（同一レース内は同値）。

    Args:
        df: ``race_id`` と ``POOL_COL`` を持つ DataFrame

    Returns:
        pd.Series: race_id を index とする pool_1m（円）
    """
    return df.groupby('race_id')[POOL_COL].first()


def apply_pool_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """``pool_1m`` がしきい値未満のレースを丸ごと除外する（spec §5.3）。

    Args:
        df: ``POOL_COL`` を持つ DataFrame
        threshold: しきい値（円）。この値以上のレースだけを残す

    Returns:
        pd.DataFrame: フィルタ通過行のみ（index は振り直す）
    """
    keep = df[POOL_COL] >= threshold
    n_races_before = int(df['race_id'].nunique())
    out = df[keep].reset_index(drop=True)
    n_races_after = int(out['race_id'].nunique()) if len(out) else 0
    ratio = (n_races_after / n_races_before * 100) if n_races_before else 0.0
    logger.info(
        'pool_1m フィルタ（>= %s円）: %s / %s レース通過（%.1f%%）',
        f'{threshold:,.0f}', f'{n_races_after:,}', f'{n_races_before:,}', ratio,
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


def describe_pool_quantiles(pool_by_race: pd.Series) -> List[Tuple[str, float]]:
    """レース総投票額の分位点一覧を作る（spec §5.2 の表と同じ並び）。

    Args:
        pool_by_race: レース単位の ``pool_1m``（円）

    Returns:
        List[Tuple[str, float]]: (分位点ラベル, 金額) の一覧
    """
    return [
        (f'q{q:.2f}', pool_threshold(pool_by_race, q))
        for q in (0.0, 0.25, 0.50, 0.75, 0.90, 1.0)
    ]


def takeout_rate() -> float:
    """票数逆算に使っている JRA 控除率を返す（ログ・検証用）。

    Returns:
        float: 控除率（単勝 0.20）
    """
    return TAKEOUT_RATE
