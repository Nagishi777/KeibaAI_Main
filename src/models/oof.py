"""専門家（base learner）ごとの OOF 予測を生成する.

Meta Model は**必ず OOF 予測**で学習しなければならない。in-sample の予測で
学習すると、Meta が「base model がその行を覚えている」度合いを学習してしまい、
本番で成立しない重みを付ける。

フォールド分割は ``LightGBMOptimizer._generate_cv_folds``（expanding walk-forward）を
そのまま使う。各フォールドは以下の3分割で、**test は予測にのみ使う**:

    fit   : 学習
    valid : early stopping とキャリブレーション（``_split_train_valid_periods``）
    test  : OOF 予測（Meta の学習データになる）

valid を train の末尾から切り出すのは、test でキャリブレートすると
Meta が楽観的な確率を学習してしまうため。
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.features.market_probability import compute_market_proba
from src.models.experts import ExpertSpec
from src.models.odds_series import BET_ODDS_COLS, PAYOUT_COLS
from src.models.ranking_label import compute_dynamic_odds_thresholds, make_relevance_labels

logger = logging.getLogger(__name__)


def generate_oof_predictions(
    optimizer,  # LightGBMOptimizer
    df: pd.DataFrame,
    experts: Sequence[ExpertSpec],
    expert_feature_cols: dict[str, list[str]],
    ranking_label_config: dict,
    n_splits: int = 5,
    min_train_months: int = 12,
    test_window_months: int = 3,
    cv_type: str = 'expanding',
) -> list[dict]:
    """全専門家の walk-forward OOF 予測を生成する.

    Args:
        optimizer: ``LightGBMOptimizer`` インスタンス（フォールド生成と trainer を借りる）
        df: 特徴量とメタ列を持つ DataFrame（``date`` / ``race_id`` 必須）
        experts: 専門家仕様のリスト
        expert_feature_cols: ``{expert_id: 入力列}``
        ranking_label_config: LambdaRank のラベル設定
        n_splits: フォールド数
        min_train_months: 拡張窓の最小学習期間
        test_window_months: テスト窓の幅
        cv_type: 'expanding' または 'rolling'

    Returns:
        list[dict]: フォールドごとの OOF 予測。各要素は
            ``{expert_id}_proba`` / ``p_market`` / ``race_id`` / ``date`` /
            ``won1`` / ``odds_win`` / ``payout_win`` / ``valid_win`` を持つ

    Raises:
        ValueError: 有効なフォールドが1つも作れない場合
    """
    work = df.sort_values(['date', 'race_id']).reset_index(drop=True)
    month_period = work['date'].dt.to_period('M')
    all_periods = sorted(month_period.unique())

    if cv_type == 'rolling':
        folds = optimizer._generate_rolling_folds(
            all_periods, n_splits, min_train_months, test_window_months
        )
    else:
        folds = optimizer._generate_cv_folds(
            all_periods, n_splits, min_train_months, test_window_months
        )
    if not folds:
        raise ValueError(
            f'OOF 用のフォールドを生成できませんでした'
            f'（データ期間 {len(all_periods)} ヶ月 / min_train={min_train_months} '
            f'/ test_window={test_window_months}）。期間か設定を見直してください。'
        )

    fold_data: list[dict] = []
    for fold_idx, (train_periods, test_periods) in enumerate(folds):
        fold_num = fold_idx + 1
        logger.info('OOF Fold %d/%d を生成中...', fold_num, len(folds))

        fit_periods, valid_periods = optimizer._split_train_valid_periods(
            train_periods, optimizer._cv_valid_months()
        )
        fit_df = work[month_period.isin(fit_periods)].reset_index(drop=True)
        test_df = work[month_period.isin(test_periods)].reset_index(drop=True)
        if len(fit_df) == 0 or len(test_df) == 0:
            logger.warning('OOF Fold %d: データが空のためスキップします', fold_num)
            continue

        if len(valid_periods) == 0:
            logger.warning(
                'OOF Fold %d: 学習期間が短く valid ホールドアウトを確保できないため '
                'test を early stopping に使います（楽観バイアスの恐れ）',
                fold_num,
            )
            valid_df = test_df
        else:
            valid_df = work[month_period.isin(valid_periods)].reset_index(drop=True)

        entry = _build_fold_entry(test_df)
        if entry is None:
            logger.warning('OOF Fold %d: 締切前オッズが揃わずスキップします', fold_num)
            continue

        for spec in experts:
            cols = expert_feature_cols[spec.expert_id]
            proba = _train_and_predict_expert(
                optimizer, spec, cols, fit_df, valid_df, test_df,
                fold_num, ranking_label_config,
            )
            if proba is None:
                continue
            entry[f'{spec.expert_id}_proba'] = proba
            if spec.min_date is not None:
                # 学習期間の制約で棄権した行を明示する（補完はしない）
                entry[f'{spec.expert_id}_available'] = (
                    (test_df['date'] >= pd.Timestamp(spec.min_date))
                    .to_numpy().astype(float)
                )

        fold_data.append(entry)

    if not fold_data:
        raise ValueError('有効な OOF フォールドが1つも生成できませんでした')

    logger.info(
        'OOF 生成完了: %d フォールド / 合計 %s 行',
        len(fold_data), f'{sum(len(f["race_id"]) for f in fold_data):,}',
    )
    return fold_data


def _build_fold_entry(test_df: pd.DataFrame) -> Optional[dict]:
    """テストフォールドから評価用メタ列を組み立てる.

    締切前オッズが欠損する行は除外する（確定オッズで代用しない）。

    Args:
        test_df: テストフォールド

    Returns:
        dict | None: メタ列。有効行が0件なら None
    """
    bet_col = BET_ODDS_COLS['win']
    if bet_col not in test_df.columns:
        return None

    odds = test_df[bet_col].to_numpy(dtype=float)
    valid = np.isfinite(odds) & (odds > 0)
    if not valid.any():
        return None

    sub = test_df[valid].reset_index(drop=True)
    odds_v = sub[bet_col].to_numpy(dtype=float)
    fin = sub['finish_position'].to_numpy()

    payout = (
        sub[PAYOUT_COLS['win']].fillna(0.0).to_numpy(dtype=float) / 100.0
        if PAYOUT_COLS['win'] in sub.columns
        else np.zeros(len(sub))
    )

    return {
        'race_id': sub['race_id'].to_numpy(),
        'date': sub['date'].to_numpy(),
        'p_market': compute_market_proba(odds_v, sub['race_id']),
        'odds_win': odds_v,
        'payout_win': payout,
        'valid_win': np.ones(len(sub), dtype=bool),
        'won1': (~pd.isna(fin)) & (np.nan_to_num(fin, nan=-1).astype(int) == 1),
        '_valid_mask': valid,
    }


def _train_and_predict_expert(
    optimizer,
    spec: ExpertSpec,
    feature_cols: list[str],
    fit_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_num: int,
    ranking_label_config: dict,
) -> Optional[np.ndarray]:
    """1専門家をフォールド内で学習し、test の予測を返す.

    Returns:
        np.ndarray | None: test（締切前オッズ有効行のみ）の予測。学習不能なら None
    """
    trainer = optimizer._trainer
    model_name = f'_oof_{spec.expert_id}_fold{fold_num}'

    fit_use = _apply_min_date(fit_df, spec)
    if len(fit_use) == 0:
        logger.warning(
            '[%s] Fold %d: min_date=%s により学習データが0件のためスキップします',
            spec.expert_id, fold_num, spec.min_date,
        )
        return None

    bet_col = BET_ODDS_COLS['win']
    test_valid = test_df[
        np.isfinite(test_df[bet_col].to_numpy(dtype=float))
        & (test_df[bet_col].to_numpy(dtype=float) > 0)
    ].reset_index(drop=True)

    try:
        if spec.objective == 'lambdarank':
            proba = _train_ranking_expert(
                trainer, model_name, feature_cols, fit_use, valid_df,
                test_valid, ranking_label_config,
            )
        else:
            proba = _train_binary_expert(
                optimizer, trainer, model_name, spec, feature_cols,
                fit_use, valid_df, test_valid,
            )
    finally:
        trainer.models.pop(model_name, None)
        trainer.calibrators.pop(model_name, None)
        trainer._init_score_models.discard(model_name)

    return proba


def _apply_min_date(df: pd.DataFrame, spec: ExpertSpec) -> pd.DataFrame:
    """専門家のデータ可用性制約を適用する."""
    if spec.min_date is None:
        return df
    return df[df['date'] >= pd.Timestamp(spec.min_date)].reset_index(drop=True)


def _train_binary_expert(
    optimizer,
    trainer,
    model_name: str,
    spec: ExpertSpec,
    feature_cols: list[str],
    fit_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """バイナリ（通常 / 残差）専門家を学習して予測する."""
    if spec.use_market_init_score:
        # 残差学習では全行に init_score = logit(p_market) が必要になる。
        # 締切前オッズが欠損する行は市場確率を作れないため学習から除外する
        # （補完すると市場アンカーが偽装される）。test 側は呼び出し元で
        # 除外済みなので fit / valid のみ絞る。
        fit_df = _drop_missing_bet_odds(fit_df)
        valid_df = _drop_missing_bet_odds(valid_df)
        if len(fit_df) == 0 or len(valid_df) == 0:
            logger.warning(
                '[%s] 締切前オッズ有効行が0件のためスキップします', spec.expert_id
            )
            return None

    y_fit = fit_df[spec.target_col]
    if y_fit.sum() == 0:
        return None

    decay = optimizer.config.get('model', {}).get('time_decay', {})
    sw = trainer.compute_time_decay_weights(
        fit_df['date'],
        recent_years=float(decay.get('recent_years', 2.0)),
        mid_years=float(decay.get('mid_years', 4.0)),
        w_recent=float(decay.get('w_recent', 3.0)),
        w_mid=float(decay.get('w_mid', 2.0)),
    )

    init_fit = init_val = init_test = None
    if spec.use_market_init_score:
        from src.features.market_probability import safe_logit

        init_fit = safe_logit(_market_proba_of(fit_df))
        init_val = safe_logit(_market_proba_of(valid_df))
        init_test = safe_logit(_market_proba_of(test_df))

    trainer.train(
        fit_df[feature_cols], y_fit,
        valid_df[feature_cols], valid_df[spec.target_col],
        model_name=model_name, sample_weight=sw,
        init_score=init_fit, init_score_val=init_val,
    )

    # キャリブレーションは valid で fit する（test は予測にのみ使う）
    if spec.calibrate and valid_df[spec.target_col].sum() > 0 and init_val is None:
        trainer.calibrate_model(
            model_name, valid_df[feature_cols], valid_df[spec.target_col],
            method='isotonic',
        )

    return trainer.predict(
        test_df[feature_cols], model_name=model_name, init_score=init_test
    )


def _drop_missing_bet_odds(df: pd.DataFrame) -> pd.DataFrame:
    """締切前オッズが欠損・非正の行を除外する.

    Args:
        df: 締切前オッズ列を持つ DataFrame

    Returns:
        pd.DataFrame: 有効行のみ（インデックスは振り直す）
    """
    odds = df[BET_ODDS_COLS['win']].to_numpy(dtype=float)
    valid = np.isfinite(odds) & (odds > 0)
    return df[valid].reset_index(drop=True)


def _market_proba_of(df: pd.DataFrame) -> np.ndarray:
    """DataFrame から市場確率を計算する（締切前オッズ欠損行は除外済み前提）."""
    bet_col = BET_ODDS_COLS['win']
    odds = df[bet_col].to_numpy(dtype=float)
    # init_score は全行に必要なため、欠損はレース平均ではなく
    # 「市場情報なし」を表す一様確率で埋めず、明示的に停止させる
    return compute_market_proba(odds, df['race_id'], validate_overround=False)


def _train_ranking_expert(
    trainer,
    model_name: str,
    feature_cols: list[str],
    fit_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ranking_label_config: dict,
) -> Optional[np.ndarray]:
    """ランキング専門家を学習し、レース単位 softmax 確率を返す."""
    from src.predict.blend_ensemble import ranking_to_proba_per_race

    scheme = ranking_label_config.get('scheme', 'positional')
    thr = None
    if scheme == 'odds_weighted_dynamic' and 'odds_win' in fit_df.columns:
        thr = compute_dynamic_odds_thresholds(fit_df)

    y_fit = pd.Series(
        make_relevance_labels(fit_df, ranking_label_config, thr), index=fit_df.index
    )
    y_val = pd.Series(
        make_relevance_labels(valid_df, ranking_label_config, thr), index=valid_df.index
    )
    groups_fit = fit_df.groupby('race_id', sort=False).size().tolist()
    groups_val = valid_df.groupby('race_id', sort=False).size().tolist()

    trainer.train_ranking(
        fit_df[feature_cols], y_fit, groups_fit,
        valid_df[feature_cols], y_val, groups_val,
        model_name=model_name,
    )
    scores = trainer.predict(test_df[feature_cols], model_name=model_name)
    # ランキングスコアは確率ではないため、レース単位 softmax で確率化してから
    # Meta に渡す（フォールド全体に softmax をかけると寄与が 1/N に潰れる）
    return ranking_to_proba_per_race(scores, test_df['race_id'], temperature=1.5)
