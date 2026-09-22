"""スタッキング（Meta Model）の学習・予測・永続化.

``docs/report/20260811_market_edge_analysis.md`` §5 P1 の
「市場アンカー型の残差学習」を Meta Model として実装する。

    logit(p_final) = β0 + β1·logit(p_market) + β2·logit(p_model) + ...

**なぜ LightGBM ではなくロジスティック回帰か**:
    OOF 行数は CV のテスト窓に律速され、ROI の実効標本単位は馬ではなく
    **レース**（同一レース内の馬は強従属）。数千レースに対して
    LightGBM Meta は base model の OOF 癖を過学習する。
    入力6本以下のロジットなら過学習しない。

**なぜ生のコンテキスト特徴量（馬場・クラス等）を入れないか**:
    §9.2 は、乖離バケット×属性の142セルを探索した結果、
    ホールドアウトで生き残ったセルがノイズ期待値を**下回った**ことを示した。
    Meta に生コンテキストを入れると「函館では E1 を信じる」のような
    セグメント探索を連続値で再現することになり、同じ罠に落ちる。
    コンテキストは E5a の内部（木構造で正則化され1モデルとして評価される）に置く。
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.market_probability import safe_logit

logger = logging.getLogger(__name__)

#: Meta Model の入力に必ず含める市場アンカー列
MARKET_ANCHOR_COL = 'logit_p_market'

#: 正則化強度の候補（nested CV で選ぶ）。相関した logit 入力に対し
#: sklearn 既定の C=1.0 は正則化が弱すぎる。
DEFAULT_C_CANDIDATES: tuple[float, ...] = (0.01, 0.1, 1.0)


def build_stack_oof_frame(
    fold_data: list[dict],
    expert_ids: Sequence[str],
) -> pd.DataFrame:
    """OOF フォールドデータを Meta Model の学習用 DataFrame に変換する.

    Args:
        fold_data: ``generate_oof_predictions`` が返すフォールドのリスト。
            各要素は ``{expert_id}_proba`` / ``p_market`` / ``race_id`` /
            ``odds_win`` / ``payout_win`` / ``valid_win`` / ``won1`` を持つ
        expert_ids: Meta の入力にする専門家ID

    Returns:
        pd.DataFrame: 1行=1出走馬。logit 変換済みの専門家出力と評価用メタ列

    Raises:
        ValueError: フォールドが空、または必要な列が欠けている場合
    """
    if not fold_data:
        raise ValueError(
            'OOF フォールドデータが空です。CV の設定（期間・n_splits）を確認してください。'
        )

    frames = []
    for i, fd in enumerate(fold_data):
        missing = [f'{e}_proba' for e in expert_ids if f'{e}_proba' not in fd]
        if missing:
            raise ValueError(
                f'フォールド {i} に専門家の予測が不足しています: {missing}'
            )
        if 'p_market' not in fd:
            raise ValueError(f'フォールド {i} に p_market がありません')

        cols: dict[str, np.ndarray] = {
            MARKET_ANCHOR_COL: safe_logit(fd['p_market']),
        }
        for e in expert_ids:
            cols[f'logit_{e}'] = safe_logit(fd[f'{e}_proba'])
            avail_key = f'{e}_available'
            if avail_key in fd:
                cols[avail_key] = np.asarray(fd[avail_key], dtype=float)

        cols['race_id'] = np.asarray(fd['race_id'])
        cols['target_win'] = np.asarray(fd['won1']).astype(int)
        for meta_col in ('odds_win', 'payout_win', 'valid_win'):
            if meta_col in fd:
                cols[meta_col] = np.asarray(fd[meta_col])
        cols['fold'] = np.full(len(fd['race_id']), i, dtype=int)
        frames.append(pd.DataFrame(cols))

    out = pd.concat(frames, ignore_index=True)
    logger.info(
        'OOF データセット: %s 行 / %s レース / %s フォールド',
        f'{len(out):,}',
        f'{out["race_id"].nunique():,}',
        out['fold'].nunique(),
    )
    return out


def meta_feature_cols(expert_ids: Sequence[str], oof_df: pd.DataFrame) -> list[str]:
    """Meta Model の入力列を決める（市場アンカー + 各専門家の logit）.

    Args:
        expert_ids: 専門家ID
        oof_df: OOF DataFrame

    Returns:
        list[str]: 入力列名
    """
    cols = [MARKET_ANCHOR_COL]
    for e in expert_ids:
        cols.append(f'logit_{e}')
        if f'{e}_available' in oof_df.columns:
            cols.append(f'{e}_available')
    return [c for c in cols if c in oof_df.columns]


def train_stack_meta(
    oof_df: pd.DataFrame,
    expert_ids: Sequence[str],
    c_candidates: Sequence[float] = DEFAULT_C_CANDIDATES,
    seed: int = 42,
) -> tuple[Pipeline, list[str], dict]:
    """OOF 予測から Meta Model（条件付きロジット）を学習する.

    正則化強度 ``C`` は**レース単位**のグループ分割で選ぶ。
    馬単位で分割すると同一レースの馬が train/valid に跨り、
    楽観的な評価になるため。

    Args:
        oof_df: ``build_stack_oof_frame`` の戻り値
        expert_ids: Meta の入力にする専門家ID
        c_candidates: 正則化強度の候補
        seed: 乱数シード

    Returns:
        tuple: (学習済み Pipeline, 入力列名, 学習情報のdict)

    Raises:
        ValueError: OOF が空、または正例が0件の場合
    """
    if oof_df.empty:
        raise ValueError('OOF DataFrame が空です')

    cols = meta_feature_cols(expert_ids, oof_df)
    X = oof_df[cols].to_numpy(dtype=float)
    y = oof_df['target_win'].to_numpy(dtype=int)

    if y.sum() == 0:
        raise ValueError('OOF に正例（1着）が1件もありません')
    if not np.isfinite(X).all():
        n_bad = int((~np.isfinite(X)).sum())
        raise ValueError(
            f'Meta 入力に非有限値が {n_bad} 個あります。'
            'logit 変換前の確率にNaN/0/1が混入していないか確認してください。'
        )

    best_c, best_score = _select_c_by_race_cv(X, y, oof_df['race_id'], c_candidates, seed)

    model = _make_pipeline(best_c, seed)
    model.fit(X, y)

    coefs = dict(zip(cols, model.named_steps['clf'].coef_[0].round(4)))
    info = {
        'C': best_c,
        'cv_logloss': round(best_score, 6),
        'n_oof_rows': int(len(oof_df)),
        'n_oof_races': int(oof_df['race_id'].nunique()),
        'pos_rate': round(float(y.mean()), 5),
        'coefficients': {k: float(v) for k, v in coefs.items()},
    }
    logger.info(
        'Meta Model 学習完了: C=%s n=%s(%sレース) 係数=%s',
        best_c, f'{len(oof_df):,}', f'{info["n_oof_races"]:,}', coefs,
    )
    return model, cols, info


def _make_pipeline(c: float, seed: int) -> Pipeline:
    """標準化 + ロジスティック回帰のパイプラインを作る."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=c, max_iter=1000, random_state=seed)),
    ])


def _select_c_by_race_cv(
    X: np.ndarray,
    y: np.ndarray,
    race_ids: pd.Series,
    c_candidates: Sequence[float],
    seed: int,
    n_splits: int = 3,
) -> tuple[float, float]:
    """レース単位のグループ分割で正則化強度を選ぶ.

    Args:
        X: 入力行列
        y: ラベル
        race_ids: レースID（グループ分割の単位）
        c_candidates: 候補
        seed: 乱数シード
        n_splits: 分割数

    Returns:
        tuple: (最良C, そのときの平均logloss)
    """
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupKFold

    if len(c_candidates) == 1:
        return float(c_candidates[0]), float('nan')

    groups = race_ids.to_numpy()
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return float(c_candidates[0]), float('nan')

    gkf = GroupKFold(n_splits=splits)
    best_c, best_score = float(c_candidates[0]), float('inf')
    for c in c_candidates:
        scores = []
        for tr, va in gkf.split(X, y, groups):
            if y[tr].sum() == 0 or y[va].sum() == 0:
                continue
            m = _make_pipeline(float(c), seed)
            m.fit(X[tr], y[tr])
            scores.append(log_loss(y[va], m.predict_proba(X[va])[:, 1], labels=[0, 1]))
        if not scores:
            continue
        mean_score = float(np.mean(scores))
        logger.info('  C=%-6s logloss=%.6f', c, mean_score)
        if mean_score < best_score:
            best_c, best_score = float(c), mean_score
    return best_c, best_score


def predict_stack(
    model: Pipeline,
    feature_cols: Sequence[str],
    expert_probas: dict[str, np.ndarray],
    p_market: np.ndarray,
    race_ids: pd.Series | np.ndarray,
    normalize_per_race: bool = True,
) -> np.ndarray:
    """Meta Model で最終確率を予測する.

    Args:
        model: ``train_stack_meta`` が返した Pipeline
        feature_cols: 学習時の入力列名（順序を保持すること）
        expert_probas: ``{expert_id: 確率配列}``
        p_market: 市場確率
        race_ids: 各行のレースID
        normalize_per_race: True ならレース内で合計1に正規化する
            （EV 比較のために必要）

    Returns:
        np.ndarray: 最終確率

    Raises:
        ValueError: 学習時の入力列を構成できない場合
    """
    available: dict[str, np.ndarray] = {MARKET_ANCHOR_COL: safe_logit(p_market)}
    for eid, proba in expert_probas.items():
        available[f'logit_{eid}'] = safe_logit(proba)

    missing = [c for c in feature_cols if c not in available]
    if missing:
        raise ValueError(
            f'Meta Model の入力列を構成できません（不足: {missing}）。'
            f'学習時の列: {list(feature_cols)} / 与えられた専門家: {sorted(expert_probas)}'
        )

    X = np.column_stack([available[c] for c in feature_cols])
    proba = model.predict_proba(X)[:, 1]

    if normalize_per_race:
        proba = normalize_within_race(proba, race_ids)
    return proba


def normalize_within_race(
    proba: np.ndarray, race_ids: pd.Series | np.ndarray
) -> np.ndarray:
    """レース内で確率の合計が1になるよう正規化する.

    Meta Model の出力は馬ごとに独立なのでレース内合計は1にならない。
    EV = 確率 × オッズ の比較可能性のために正規化する。

    Args:
        proba: 確率
        race_ids: 各行のレースID

    Returns:
        np.ndarray: レースごとに合計1の確率
    """
    s = pd.Series(np.asarray(proba, dtype=float))
    groups = pd.Series(
        race_ids.to_numpy() if hasattr(race_ids, 'to_numpy') else np.asarray(race_ids)
    )
    denom = s.groupby(groups).transform('sum')
    # レース内合計が0（全馬0確率）は異常。補完せず停止する
    if (denom <= 0).any():
        raise ValueError(
            'レース内の確率合計が0のレースがあります。'
            'Meta Model の出力を確認してください（補完はしません）。'
        )
    return (s / denom).to_numpy()


def save_stack(
    model_dir: Path,
    model: Pipeline,
    feature_cols: Sequence[str],
    expert_ids: Sequence[str],
    info: dict,
) -> Path:
    """Meta Model と付随情報を保存する.

    Args:
        model_dir: ``data/model`` 相当のディレクトリ
        model: 学習済み Pipeline
        feature_cols: 入力列名（順序込み）
        expert_ids: 専門家ID
        info: 学習情報

    Returns:
        Path: 保存した pickle のパス
    """
    stack_dir = Path(model_dir) / 'stack'
    stack_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = stack_dir / 'meta_model.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(model, f)

    payload = {
        'meta_feature_cols': list(feature_cols),
        'expert_ids': list(expert_ids),
        **info,
    }
    with open(stack_dir / 'meta_model.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info('Meta Model を保存: %s', pkl_path)
    return pkl_path


def load_stack(model_dir: Path) -> tuple[Pipeline, list[str], list[str], dict]:
    """保存した Meta Model を読み込む.

    Args:
        model_dir: ``data/model`` 相当のディレクトリ

    Returns:
        tuple: (Pipeline, 入力列名, 専門家ID, 学習情報)

    Raises:
        FileNotFoundError: 未学習の場合
    """
    stack_dir = Path(model_dir) / 'stack'
    pkl_path = stack_dir / 'meta_model.pkl'
    json_path = stack_dir / 'meta_model.json'
    if not pkl_path.exists() or not json_path.exists():
        raise FileNotFoundError(
            f'Meta Model が見つかりません: {pkl_path}. '
            '先にスタック学習を実行してください'
        )
    with open(pkl_path, 'rb') as f:
        model = pickle.load(f)
    with open(json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    return (
        model,
        list(payload['meta_feature_cols']),
        list(payload['expert_ids']),
        payload,
    )


def assert_no_period_overlap(
    oof_df: pd.DataFrame,
    selection_fit_start: str,
    selection_fit_end: str,
    date_col: Optional[pd.Series] = None,
) -> None:
    """OOF 期間と選抜バイアス補正の fit 期間が重ならないことを検証する.

    Meta Model は OOF（モデル学習期間内のCVテスト窓）で学習し、
    選抜バイアス補正は 2024-07〜12 の学習期間外データで fit する。
    両者が重なると二重補正になるため、慣習に委ねずコードで検証する。

    Args:
        oof_df: OOF DataFrame
        selection_fit_start: 選抜バイアス補正の fit 開始日
        selection_fit_end: 同 終了日
        date_col: OOF の日付列（``oof_df`` に 'date' が無い場合に渡す）

    Raises:
        ValueError: 期間が重なる場合
    """
    dates = date_col if date_col is not None else oof_df.get('date')
    if dates is None:
        logger.warning(
            'OOF に日付列が無いため期間重複の検証をスキップしました。'
            '呼び出し側で date_col を渡すことを推奨します。'
        )
        return

    oof_min, oof_max = pd.Timestamp(dates.min()), pd.Timestamp(dates.max())
    sel_start = pd.Timestamp(selection_fit_start)
    sel_end = pd.Timestamp(selection_fit_end)
    if oof_max >= sel_start and oof_min <= sel_end:
        raise ValueError(
            f'OOF 期間（{oof_min.date()}〜{oof_max.date()}）と'
            f'選抜バイアス補正の fit 期間（{sel_start.date()}〜{sel_end.date()}）が'
            '重なっています。二重補正になるため停止します。'
        )
