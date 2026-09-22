"""
馬連（umaren）ペアデータ構築モジュール

馬単位（1行1頭）の特徴量 DataFrame から、馬連の学習・評価に使う
ペア単位（1行1組み合わせ）の DataFrame を構築する。

設計上の要点:
    - **順序不変性**: ペア特徴量は ``max`` / ``min`` / ``sum`` / ``absdiff`` で集約し、
      ``(a, b)`` と ``(b, a)`` で必ず同じ値になるようにする。生の ``f_a`` / ``f_b`` を
      そのまま入れると「馬番が若い方」という無意味な情報を学習してしまう。
    - **候補の絞り込み**: 全ペアを使うと正例率が 1/C(n,2) ≒ 0.8% と極端に不均衡になるため、
      単勝モデルの予測確率上位 K 頭のペアのみを対象にする。
    - **リーク防止**: 絞り込みに使う単勝確率は、当該レースを学習に含まないモデルの
      出力でなければならない（呼び出し側の責務）。
"""
import logging
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ペア特徴量の集約方法（順序不変なものだけを許可する）
SUPPORTED_AGGREGATIONS = ('max', 'min', 'sum', 'absdiff')

# ペア行を一意に決めるキー
PAIR_KEYS = ('race_id', 'horse_number_a', 'horse_number_b')

# ペア単位で持ち回すメタ列（特徴量ではない）
PAIR_META_COLS = (
    'race_id', 'date', 'horse_number_a', 'horse_number_b', 'target_umaren',
)


def build_umaren_pairs(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    win_proba: Optional[np.ndarray] = None,
    top_k_horses: int = 8,
    aggregations: Sequence[str] = ('max', 'min', 'sum', 'absdiff'),
    ranking_scores: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """馬単位 DataFrame から馬連ペア DataFrame を構築する。

    Args:
        df: 馬単位 DataFrame（``race_id`` / ``horse_number`` / ``date`` 必須。
            ラベル生成には ``finish_position`` が必要）
        feature_cols: ペアに展開する馬単位の特徴量カラム
        win_proba: 各行の単勝予測確率（df と同じ行数・行順）。
            指定すると上位 ``top_k_horses`` 頭に候補を絞り、
            ペア専用特徴量（確率の積・Harville）も生成する。
        top_k_horses: 候補として残すレースあたりの頭数（0 で絞り込みなし）
        aggregations: ペア集約方法（``SUPPORTED_AGGREGATIONS`` の部分集合）
        ranking_scores: ランキングモデルのスコア（df と同順、省略可）。
            指定するとレース内順位の和・差をペア特徴量に加える。

    Returns:
        pd.DataFrame: ペア単位 DataFrame。
            ``race_id`` / ``date`` / ``horse_number_a`` / ``horse_number_b`` /
            ``target_umaren`` と、ペア特徴量カラムを持つ。

    Raises:
        ValueError: 必須列が無い場合、または不正な集約方法が指定された場合
    """
    required = ('race_id', 'horse_number', 'date')
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"ペア構築に必要な列がありません: {missing}")

    bad_agg = [a for a in aggregations if a not in SUPPORTED_AGGREGATIONS]
    if bad_agg:
        raise ValueError(
            f"サポートされていない集約方法: {bad_agg}"
            f"（{list(SUPPORTED_AGGREGATIONS)} のいずれか。順序不変性のため他は許可しない）"
        )
    if not aggregations:
        raise ValueError("aggregations が空です。1つ以上指定してください。")

    work = df.reset_index(drop=True).copy()
    if win_proba is not None:
        if len(win_proba) != len(work):
            raise ValueError(
                f"win_proba の長さが df と一致しません: {len(win_proba)} != {len(work)}"
            )
        work['_win_proba'] = np.asarray(win_proba, dtype=float)
    if ranking_scores is not None:
        if len(ranking_scores) != len(work):
            raise ValueError(
                f"ranking_scores の長さが df と一致しません: {len(ranking_scores)} != {len(work)}"
            )
        work['_ranking_score'] = np.asarray(ranking_scores, dtype=float)

    # 候補の絞り込み（単勝確率上位 K 頭）
    if win_proba is not None and top_k_horses > 0:
        rank = work.groupby('race_id')['_win_proba'].rank(ascending=False, method='first')
        n_before = len(work)
        work = work[rank <= top_k_horses].reset_index(drop=True)
        logger.info(
            f"馬連候補を単勝確率上位 {top_k_horses} 頭に絞り込み: {n_before} → {len(work)} 行"
        )

    pairs = _expand_pairs(work)
    if pairs.empty:
        raise ValueError("馬連ペアが1件も生成されませんでした。入力データを確認してください。")

    pair_features = _aggregate_pair_features(work, pairs, feature_cols, aggregations)
    out = pd.concat([pairs.reset_index(drop=True), pair_features], axis=1)

    if win_proba is not None:
        out = _add_win_proba_pair_features(out, work)
    if ranking_scores is not None:
        out = _add_ranking_pair_features(out, work)
    if 'popularity' in work.columns:
        out = _add_scalar_pair_features(out, work, 'popularity', 'pair_popularity')

    logger.info(
        f"馬連ペア構築完了: {len(out)} 行 / {out['race_id'].nunique()} レース "
        f"（1レース平均 {len(out) / max(out['race_id'].nunique(), 1):.1f} ペア）"
    )
    return out


def _expand_pairs(work: pd.DataFrame) -> pd.DataFrame:
    """レースごとに全ての馬番ペア（a < b）を展開する。

    Args:
        work: 馬単位 DataFrame（絞り込み済み）

    Returns:
        pd.DataFrame: [race_id, date, horse_number_a, horse_number_b, _idx_a, _idx_b]
            ``_idx_a`` / ``_idx_b`` は work の行インデックス
    """
    records: List[Dict] = []
    for race_id, race in work.groupby('race_id', sort=False):
        # 馬番昇順に並べてから組み合わせを取るので必ず a < b になる
        race_sorted = race.sort_values('horse_number')
        numbers = race_sorted['horse_number'].to_numpy()
        indices = race_sorted.index.to_numpy()
        race_date = race_sorted['date'].iloc[0]
        for (ia, na), (ib, nb) in combinations(zip(indices, numbers), 2):
            records.append({
                'race_id': race_id,
                'date': race_date,
                'horse_number_a': int(na),
                'horse_number_b': int(nb),
                '_idx_a': int(ia),
                '_idx_b': int(ib),
            })
    return pd.DataFrame.from_records(records)


def _aggregate_pair_features(
    work: pd.DataFrame,
    pairs: pd.DataFrame,
    feature_cols: Sequence[str],
    aggregations: Sequence[str],
) -> pd.DataFrame:
    """馬単位特徴量を順序不変なペア特徴量に集約する。

    Args:
        work: 馬単位 DataFrame
        pairs: ``_idx_a`` / ``_idx_b`` を持つペア DataFrame
        feature_cols: 集約対象の特徴量カラム
        aggregations: 'max' / 'min' / 'sum' / 'absdiff'

    Returns:
        pd.DataFrame: ペア特徴量（列名は ``{feature}_{agg}``）
    """
    idx_a = pairs['_idx_a'].to_numpy()
    idx_b = pairs['_idx_b'].to_numpy()

    # カテゴリ列はペアで同一（レース共通）なのでそのまま1列として残す。
    # 数値列のみを max/min/sum/absdiff で集約する。
    numeric_cols = [
        c for c in feature_cols
        if c in work.columns and pd.api.types.is_numeric_dtype(work[c])
    ]
    race_level_cols = [
        c for c in feature_cols
        if c in work.columns and c not in numeric_cols
    ]

    out: Dict[str, np.ndarray] = {}
    for col in numeric_cols:
        values = work[col].to_numpy(dtype=float)
        va, vb = values[idx_a], values[idx_b]
        for agg in aggregations:
            if agg == 'max':
                out[f'{col}_max'] = np.maximum(va, vb)
            elif agg == 'min':
                out[f'{col}_min'] = np.minimum(va, vb)
            elif agg == 'sum':
                out[f'{col}_sum'] = va + vb
            else:  # absdiff
                out[f'{col}_absdiff'] = np.abs(va - vb)

    result = pd.DataFrame(out)
    # レース共通の非数値列（カテゴリ dtype 等）は a 側の値をそのまま採用（ペア内で同一）。
    # work[col].to_numpy()[idx_a] だと pandas の category dtype が失われ通常の
    # object/str に変換されてしまい、LightGBM が「非数値・非カテゴリ列」として
    # 拒否する（ValueError: pandas dtypes must be int, float or bool）。
    # .iloc[idx_a] なら Series のまま抽出できるため category dtype を保持できる。
    for col in race_level_cols:
        result[col] = work[col].iloc[idx_a].reset_index(drop=True)
    return result


def _add_scalar_pair_features(
    out: pd.DataFrame, work: pd.DataFrame, col: str, prefix: str
) -> pd.DataFrame:
    """単一の馬単位列から順序不変なペア特徴量（和・差）を追加する。

    Args:
        out: ペア DataFrame（``_idx_a`` / ``_idx_b`` を持つ）
        work: 馬単位 DataFrame
        col: 対象の馬単位列名
        prefix: 生成する列名のプレフィックス

    Returns:
        pd.DataFrame: 特徴量を追加した DataFrame
    """
    values = pd.to_numeric(work[col], errors='coerce').to_numpy(dtype=float)
    va = values[out['_idx_a'].to_numpy()]
    vb = values[out['_idx_b'].to_numpy()]
    out[f'{prefix}_sum'] = va + vb
    out[f'{prefix}_absdiff'] = np.abs(va - vb)
    return out


def _add_win_proba_pair_features(out: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """単勝予測確率からペア専用特徴量を生成する。

    - ``pair_win_proba_product``: 両馬の勝率の積（独立近似）
    - ``pair_win_proba_sum`` / ``pair_win_proba_absdiff``: 和・差
    - ``pair_win_proba_harville``: Harville 公式による「両馬が1着・2着を占める」確率

    Args:
        out: ペア DataFrame（``_idx_a`` / ``_idx_b`` を持つ）
        work: ``_win_proba`` を持つ馬単位 DataFrame

    Returns:
        pd.DataFrame: 特徴量を追加した DataFrame
    """
    proba = work['_win_proba'].to_numpy(dtype=float)
    idx_a = out['_idx_a'].to_numpy()
    idx_b = out['_idx_b'].to_numpy()
    pa, pb = proba[idx_a], proba[idx_b]

    out['pair_win_proba_product'] = pa * pb
    out['pair_win_proba_sum'] = pa + pb
    out['pair_win_proba_absdiff'] = np.abs(pa - pb)

    # Harville: レース内で確率を正規化した強さ s から
    #   P(A,B が1-2着) = s_a*s_b/(1-s_a) + s_b*s_a/(1-s_b)
    race_sum = work.groupby('race_id')['_win_proba'].transform('sum').to_numpy(dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        strength = np.where(race_sum > 0, proba / race_sum, 0.0)
    sa, sb = strength[idx_a], strength[idx_b]
    denom_a = 1.0 - sa
    denom_b = 1.0 - sb
    with np.errstate(divide='ignore', invalid='ignore'):
        term_a = np.where(denom_a > 1e-12, sa * sb / denom_a, 0.0)
        term_b = np.where(denom_b > 1e-12, sb * sa / denom_b, 0.0)
    out['pair_win_proba_harville'] = np.clip(
        np.nan_to_num(term_a + term_b), 0.0, 1.0
    )
    return out


def _add_ranking_pair_features(out: pd.DataFrame, work: pd.DataFrame) -> pd.DataFrame:
    """ランキングモデルのレース内順位からペア特徴量を生成する。

    Args:
        out: ペア DataFrame（``_idx_a`` / ``_idx_b`` を持つ）
        work: ``_ranking_score`` を持つ馬単位 DataFrame

    Returns:
        pd.DataFrame: 特徴量を追加した DataFrame
    """
    rank = (
        work.groupby('race_id')['_ranking_score']
        .rank(ascending=False, method='min')
        .to_numpy(dtype=float)
    )
    ra = rank[out['_idx_a'].to_numpy()]
    rb = rank[out['_idx_b'].to_numpy()]
    out['pair_rank_sum'] = ra + rb
    out['pair_rank_absdiff'] = np.abs(ra - rb)
    out['pair_rank_max'] = np.maximum(ra, rb)
    return out


def add_umaren_labels(
    pairs: pd.DataFrame,
    payout_umaren: pd.DataFrame,
) -> pd.DataFrame:
    """確定馬連払戻の的中組み合わせから ``target_umaren`` を付与する。

    的中ラベルは ``finish_position`` ではなく**払戻データの的中組み合わせ**を正とする。
    同着などの特殊ケースで払戻ルールと着順が食い違うため、払戻と整合させる。

    払戻データに存在しないレース（＝的中組み合わせが不明なレース）は、
    ラベルを 0 で埋めると「全ペア不的中」という誤ったラベルになるため、
    補完せず**レースごと除外**し、除外件数をログに残す。

    Args:
        pairs: ペア DataFrame（``race_id`` / ``horse_number_a`` / ``horse_number_b``）
        payout_umaren: ``src.models.odds_series.build_umaren_payout`` の戻り値

    Returns:
        pd.DataFrame: ``target_umaren`` を付与し、払戻不明レースを除外した DataFrame

    Raises:
        ValueError: 有効なレースが1件も残らない場合
    """
    keys = list(PAIR_KEYS)
    hit = payout_umaren[keys].copy()
    hit['target_umaren'] = 1

    out = pairs.merge(hit, on=keys, how='left')
    out['target_umaren'] = out['target_umaren'].fillna(0).astype(int)

    # 払戻データに的中組み合わせがあるレースのみを有効とする
    known_races = set(payout_umaren['race_id'].unique())
    valid = out['race_id'].isin(known_races)
    n_dropped_races = int(out.loc[~valid, 'race_id'].nunique())
    if n_dropped_races > 0:
        logger.warning(
            f"馬連の確定払戻が存在しない {n_dropped_races} レース"
            f"（{int((~valid).sum())} ペア）を学習・評価対象外にしました"
        )
    out = out[valid].reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "馬連の的中ラベルを付与できるレースが0件です。"
            "払戻データ（umaren_a / umaren_b）との race_id の対応を確認してください。"
        )

    pos_rate = float(out['target_umaren'].mean())
    logger.info(
        f"馬連ラベル付与完了: {len(out)} ペア / {out['race_id'].nunique()} レース、"
        f"正例率 {pos_rate:.4f}"
    )
    return out


def pair_feature_cols(pairs: pd.DataFrame) -> List[str]:
    """ペア DataFrame からモデル入力に使う特徴量カラムを抽出する。

    メタ列（``race_id`` / ``date`` / 馬番 / ラベル）、内部作業列（``_`` 始まり）、
    および評価専用のオッズ・払戻列を除外する。

    Args:
        pairs: ペア DataFrame

    Returns:
        List[str]: 特徴量カラムのリスト
    """
    from src.models.odds_series import BET_ODDS_COLS, PAYOUT_COLS

    excluded = set(PAIR_META_COLS)
    excluded.update(BET_ODDS_COLS.values())
    excluded.update(PAYOUT_COLS.values())
    return [
        c for c in pairs.columns
        if c not in excluded and not c.startswith('_')
    ]
