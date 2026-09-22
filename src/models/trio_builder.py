"""
三連複（sanfuku）3頭組み合わせデータ構築モジュール

馬単位（1行1頭）の DataFrame から、三連複の評価に使う
組み合わせ単位（1行1組）の DataFrame を構築する。
``pair_builder.py``（馬連 = 2頭）の3頭版にあたる。

設計上の要点:
    - **順序不変性**: 三連複は着順を問わないため、馬番を ``a < b < c`` に
      正規化してキーにする。
    - **候補の絞り込み**: 全組み合わせは C(18,3)=816 点と膨大なので、
      単勝モデルの予測確率上位 K 頭の組み合わせ C(K,3) のみを対象にする。
    - **確率の導出**: 専用モデルは学習せず、単勝確率から Harville 公式で
      「3頭が1〜3着を占める確率」を計算する（3! = 6 順列の和）。
      生の Harville 値は系統的に過大評価するため、必ず
      ``calibrate_trio_proba`` で較正してから EV 計算に使うこと。
    - **リーク防止**: 絞り込み・確率導出に使う単勝確率は、当該レースを
      学習に含まないモデルの出力でなければならない（呼び出し側の責務）。
      較正も学習期間で fit し評価期間に適用する。
"""
import logging
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.module.harville_calculator import calc_top3_all_proba_harville_batch

logger = logging.getLogger(__name__)

# 三連複の行を一意に決めるキー
TRIO_KEYS = ('race_id', 'horse_number_a', 'horse_number_b', 'horse_number_c')

# JRA の控除率（三連複）。合成オッズの推定に使う。
# 参考: 三連複の払戻率は 75%（＝控除率 25%）
DEFAULT_TAKEOUT_RATE = 0.25


def build_trio_combos(
    df: pd.DataFrame,
    win_proba: np.ndarray,
    top_k_horses: int = 6,
    proba_col: str = '_win_proba',
) -> pd.DataFrame:
    """レースごとに単勝確率上位 K 頭の3頭組み合わせ C(K,3) を展開する。

    Args:
        df: 馬単位 DataFrame（``race_id`` / ``horse_number`` / ``date`` 必須）
        win_proba: 各行の単勝予測確率（df と同じ行数・行順）
        top_k_horses: 候補として残すレースあたりの頭数（3 以上）
        proba_col: 内部で使う確率列名

    Returns:
        pd.DataFrame: [race_id, date, horse_number_a/b/c, trio_proba_raw,
            pair_win_proba_prod] を持つ組み合わせ単位 DataFrame

    Raises:
        ValueError: 必須列が無い、行数不一致、top_k_horses が 3 未満の場合
    """
    required = ('race_id', 'horse_number', 'date')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"三連複の組み合わせ構築に必要な列がありません: {missing}")
    if len(win_proba) != len(df):
        raise ValueError(
            f"win_proba の長さが df と一致しません: {len(win_proba)} != {len(df)}"
        )
    if top_k_horses < 3:
        raise ValueError(
            f"top_k_horses は 3 以上である必要があります（3頭組み合わせのため）: {top_k_horses}"
        )

    work = df.reset_index(drop=True).copy()
    work[proba_col] = np.asarray(win_proba, dtype=float)

    # 単勝確率上位 K 頭に絞る
    rank = work.groupby('race_id')[proba_col].rank(ascending=False, method='first')
    n_before = len(work)
    work = work[rank <= top_k_horses].reset_index(drop=True)
    logger.info(
        f"三連複候補を単勝確率上位 {top_k_horses} 頭に絞り込み: {n_before} → {len(work)} 行"
    )

    # 組み合わせインデックス C(k,3) は top_k_horses ごとに使い回せるのでキャッシュする
    combo_idx_cache: Dict[int, np.ndarray] = {}

    chunks: List[pd.DataFrame] = []
    for race_id, race in work.groupby('race_id', sort=False):
        # 馬番昇順に並べてから組み合わせを取るので必ず a < b < c になる
        race_sorted = race.sort_values('horse_number')
        n = len(race_sorted)
        if n < 3:
            continue
        numbers = race_sorted['horse_number'].to_numpy()
        race_date = race_sorted['date'].iloc[0]
        proba = race_sorted[proba_col].to_numpy(dtype=float)

        # レース内の強さを1回だけ正規化して使い回す（組み合わせごとの再計算を避ける）
        strengths = np.clip(proba, 1e-9, None)
        s = strengths / strengths.sum()

        combo_idx = combo_idx_cache.get(n)
        if combo_idx is None:
            combo_idx = np.array(list(combinations(range(n), 3)), dtype=np.intp)
            combo_idx_cache[n] = combo_idx

        trio_proba = calc_top3_all_proba_harville_batch(s, combo_idx)
        pair_prod = (
            proba[combo_idx[:, 0]] * proba[combo_idx[:, 1]] * proba[combo_idx[:, 2]]
        )

        chunks.append(pd.DataFrame({
            'race_id': race_id,
            'date': race_date,
            'horse_number_a': numbers[combo_idx[:, 0]].astype(int),
            'horse_number_b': numbers[combo_idx[:, 1]].astype(int),
            'horse_number_c': numbers[combo_idx[:, 2]].astype(int),
            'trio_proba_raw': trio_proba,
            'pair_win_proba_prod': pair_prod,
        }))

    if not chunks:
        raise ValueError(
            "三連複の組み合わせが1件も生成されませんでした。入力データを確認してください。"
        )

    out = pd.concat(chunks, ignore_index=True)
    n_races = out['race_id'].nunique()
    logger.info(
        f"三連複組み合わせ構築完了: {len(out):,} 行 / {n_races:,} レース "
        f"（1レース平均 {len(out) / max(n_races, 1):.1f} 点）"
    )
    return out


def add_sanfuku_labels(
    trios: pd.DataFrame,
    payout_sanfuku: pd.DataFrame,
) -> pd.DataFrame:
    """確定三連複払戻の的中組み合わせから ``target_sanfuku`` を付与する。

    的中ラベルは ``finish_position`` ではなく**払戻データの的中組み合わせ**を正とする。
    同着などの特殊ケースで払戻ルールと着順が食い違うため、払戻と整合させる。

    払戻データに存在しないレース（＝的中組み合わせが不明なレース）は、
    ラベルを 0 で埋めると「全点不的中」という誤ったラベルになるため、
    補完せず**レースごと除外**し、除外件数をログに残す
    （``pair_builder.add_umaren_labels`` と同じ方針）。

    Args:
        trios: 組み合わせ DataFrame（``build_trio_combos`` の戻り値）
        payout_sanfuku: ``odds_series.build_sanfuku_payout`` の戻り値

    Returns:
        pd.DataFrame: ``target_sanfuku`` / ``payout_sanfuku`` を付与し、
            払戻不明レースを除外した DataFrame

    Raises:
        ValueError: 有効なレースが1件も残らない場合
    """
    keys = list(TRIO_KEYS)
    hit = payout_sanfuku[keys + ['payout_sanfuku']].copy()
    hit['target_sanfuku'] = 1

    out = trios.merge(hit, on=keys, how='left')
    out['target_sanfuku'] = out['target_sanfuku'].fillna(0).astype(int)
    # 不的中行の払戻は 0（的中行のみ実際の払戻金が入る）
    out['payout_sanfuku'] = out['payout_sanfuku'].fillna(0.0)

    known_races = set(payout_sanfuku['race_id'].unique())
    valid = out['race_id'].isin(known_races)
    n_dropped_races = int(out.loc[~valid, 'race_id'].nunique())
    if n_dropped_races > 0:
        logger.warning(
            f"三連複の確定払戻が存在しない {n_dropped_races} レース"
            f"（{int((~valid).sum())} 点）を評価対象外にしました"
        )
    out = out[valid].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "三連複の的中ラベルを付与できるレースが0件です。"
            "払戻データ（sanfuku_a/b/c）との race_id の対応を確認してください。"
        )

    pos_rate = float(out['target_sanfuku'].mean())
    logger.info(
        f"三連複ラベル付与完了: {len(out):,} 点 / {out['race_id'].nunique():,} レース、"
        f"的中率 {pos_rate:.4f}"
    )
    return out


def fit_trio_calibrator(
    raw_proba: np.ndarray,
    target: np.ndarray,
) -> IsotonicRegression:
    """Harville 生確率を実際の的中率へ写像する較正器を学習する。

    Harville 公式は「各馬の強さが独立」という仮定を置くため、
    3頭同時的中の確率を系統的に**過大評価**する（実測で約2倍）。
    較正しないまま EV を計算すると全点で期待値がプラスに見え、
    偽の有望戦略が生まれるため必須の工程。

    IMPORTANT:
        リーク防止のため**学習期間のデータのみ**で fit すること。
        評価期間で fit すると的中率に張り付いて回収率が楽観バイアスになる。

    Args:
        raw_proba: Harville 生確率（学習期間）
        target: 的中ラベル 0/1（学習期間）

    Returns:
        IsotonicRegression: 学習済み較正器

    Raises:
        ValueError: 入力の行数が一致しない、または正例が1件も無い場合
    """
    raw = np.asarray(raw_proba, dtype=float)
    y = np.asarray(target, dtype=float)
    if len(raw) != len(y):
        raise ValueError(f"raw_proba と target の長さが一致しません: {len(raw)} != {len(y)}")
    if y.sum() == 0:
        raise ValueError(
            "較正データに的中（正例）が1件もありません。学習期間を広げてください。"
        )

    calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    calibrator.fit(raw, y)
    logger.info(
        f"三連複確率の較正器を学習: n={len(raw):,} "
        f"生確率平均 {raw.mean():.4f} → 実際の的中率 {y.mean():.4f} "
        f"（過大評価 {raw.mean() / max(y.mean(), 1e-12):.2f} 倍）"
    )
    return calibrator


def calibrate_trio_proba(
    calibrator: IsotonicRegression,
    raw_proba: np.ndarray,
) -> np.ndarray:
    """較正器を適用して Harville 生確率を補正する。

    Args:
        calibrator: ``fit_trio_calibrator`` の戻り値
        raw_proba: Harville 生確率

    Returns:
        np.ndarray: 較正済み確率（0〜1）
    """
    return np.clip(calibrator.predict(np.asarray(raw_proba, dtype=float)), 0.0, 1.0)


def market_implied_win_proba(
    df: pd.DataFrame,
    odds_col: str = 'odds_pre_win',
    race_col: str = 'race_id',
) -> np.ndarray:
    """単勝の締切前オッズから市場の含意勝率を復元する。

    オッズの逆数がその馬の含意勝率に比例する。控除率の分だけ合計が 1 を
    超えるため、レース内で正規化して確率に直す::

        含意勝率 = (1 / odds) / Σ(1 / odds)

    これは**予測時点で入手可能な市場データ**であり、確定払戻（レース結果）を
    一切含まないためリークしない。

    Args:
        df: 馬単位 DataFrame（``odds_col`` / ``race_col`` 必須）
        odds_col: 単勝締切前オッズの列名
        race_col: レースIDの列名

    Returns:
        np.ndarray: レース内で正規化された含意勝率（df と同じ行順）

    Raises:
        ValueError: 必須列が無い場合
    """
    for col in (odds_col, race_col):
        if col not in df.columns:
            raise ValueError(f"市場含意勝率の計算に必要な列がありません: {col}")

    odds = df[odds_col].to_numpy(dtype=float)
    # オッズ欠損・非正の行は寄与 0 にする（補完しない）
    inv = np.where(np.isfinite(odds) & (odds > 0), 1.0 / np.maximum(odds, 1e-12), 0.0)
    work = pd.DataFrame({race_col: df[race_col].to_numpy(), '_inv': inv})
    total = work.groupby(race_col)['_inv'].transform('sum').to_numpy()
    with np.errstate(divide='ignore', invalid='ignore'):
        proba = np.where(total > 0, inv / total, 0.0)
    return proba


def market_implied_trio_odds(
    market_trio_proba: np.ndarray,
    takeout_rate: float = DEFAULT_TAKEOUT_RATE,
    max_odds: float = 10_000.0,
) -> np.ndarray:
    """市場の含意3頭確率から賭け判断用のオッズを推定する。

    三連複には締切前オッズの実データが無いため、**単勝の締切前オッズ**という
    実在の市場データから Harville で3頭確率を導き、控除率を差し引いて
    オッズに変換する::

        推定オッズ = (1 - 控除率) / 市場含意3頭確率

    IMPORTANT (なぜ市場確率を使うのか):
        モデル自身の較正済み確率からオッズを作ると
        ``EV = p × ((1-t)/p) = 1-t`` と**全点で定数**になり、EV による選別が
        原理的に機能しない（＝賭け対象が 0 点か全点かの二択になる）。
        市場含意確率はモデルと独立した情報源なので、
        「モデルが市場より高く評価している組み合わせ」を EV で選別できる。

    この推定の限界:
        単勝オッズから合成した値であり、三連複市場の実オッズではない。
        三連複特有の売れ方（ボックス買いの偏り等）は表現できないため、
        得られる回収率は参考値である。

    Args:
        market_trio_proba: 市場含意勝率から Harville で求めた3頭確率
        takeout_rate: 控除率（三連複は 0.25）
        max_odds: オッズの上限（確率が極端に小さい行の発散を防ぐ）

    Returns:
        np.ndarray: 推定オッズ（確率 0 の行は 0 を返し、賭け対象外にする）

    Raises:
        ValueError: takeout_rate が [0, 1) の範囲外の場合
    """
    if not 0.0 <= takeout_rate < 1.0:
        raise ValueError(f"takeout_rate は 0 以上 1 未満である必要があります: {takeout_rate}")

    p = np.asarray(market_trio_proba, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        odds = (1.0 - takeout_rate) / p
    # 確率 0（＝賭けようがない）行はオッズ 0 にして下流のフィルタで落とす
    odds = np.where((p > 0) & np.isfinite(odds), odds, 0.0)
    return np.minimum(odds, max_odds)


def sweep_top_k(
    df: pd.DataFrame,
    win_proba: np.ndarray,
    payout_sanfuku: pd.DataFrame,
    k_values: Sequence[int] = (3, 4, 5, 6),
) -> Tuple[Dict[int, pd.DataFrame], pd.DataFrame]:
    """複数の top_k で組み合わせを構築し、点数・的中率を比較する。

    Args:
        df: 馬単位 DataFrame
        win_proba: 単勝予測確率（df と同じ行数・行順）
        payout_sanfuku: ``build_sanfuku_payout`` の戻り値
        k_values: 試す top_k の一覧

    Returns:
        Tuple:
            - Dict[int, pd.DataFrame]: k → ラベル付与済み組み合わせ DataFrame
            - pd.DataFrame: k ごとの要約（点数・レース数・的中率・的中レース率）
    """
    combos: Dict[int, pd.DataFrame] = {}
    rows: List[Dict] = []

    for k in k_values:
        trios = build_trio_combos(df, win_proba, top_k_horses=k)
        trios = add_sanfuku_labels(trios, payout_sanfuku)
        combos[k] = trios

        n_races = trios['race_id'].nunique()
        # 「候補の中に的中組み合わせが含まれていたレース」の割合（＝到達可能な上限）
        hit_races = int(trios.groupby('race_id')['target_sanfuku'].max().sum())
        rows.append({
            'top_k': k,
            'combos_per_race': round(len(trios) / max(n_races, 1), 1),
            'n_combos': len(trios),
            'n_races': n_races,
            'hit_rate(%)': round(trios['target_sanfuku'].mean() * 100, 3),
            'reachable_races(%)': round(hit_races / max(n_races, 1) * 100, 2),
        })

    summary = pd.DataFrame(rows).set_index('top_k')
    return combos, summary
