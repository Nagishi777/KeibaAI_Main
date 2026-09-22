"""
LightGBM 最適化モジュール

クロスバリデーション・ハイパーパラメータ最適化（Optuna）・
ブレンド係数最適化を担う。
学習・予測・保存は trainer.py の LightGBMTrainer に委譲する。
"""
import logging
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.evaluation import ModelEvaluator
from src.models.odds_series import BET_ODDS_COLS, PAYOUT_COLS, has_two_source_odds
from src.models.ranking_label import (
    compute_dynamic_odds_thresholds,
    make_relevance_labels,
)
from src.models.trainer import LightGBMTrainer

logger = logging.getLogger(__name__)


class LightGBMOptimizer:
    """
    LightGBM モデルの最適化を管理するクラス。

    責務:
        - 時系列クロスバリデーション（バイナリ / ランキング）
        - Optuna によるハイパーパラメータ最適化
        - Optuna による blend_alpha / ranking_temperature の最適化
    """

    def __init__(self, trainer: LightGBMTrainer, config: dict):
        """
        初期化

        Args:
            trainer: 学習・予測を担う LightGBMTrainer インスタンス
            config: 設定辞書（config.yaml のルート辞書）
        """
        self._trainer = trainer
        self.config = config

    # ------------------------------------------------------------------
    # フォールド生成
    # ------------------------------------------------------------------

    def _generate_cv_folds(
        self,
        all_periods: pd.PeriodIndex,
        n_splits: int,
        min_train_months: int,
        test_window_months: int
    ) -> List[Tuple]:
        """
        拡張ウィンドウ（Walk-Forward）フォールドを生成する。
        学習データを毎フォールドで拡張し、テストウィンドウは固定幅でスライドする。

        Args:
            all_periods: 月単位のユニーク期間（昇順）
            n_splits: フォールド数
            min_train_months: 最小学習期間（月数）
            test_window_months: テストウィンドウ幅（月数）

        Returns:
            List of (train_periods, test_periods) tuples
        """
        total = len(all_periods)
        max_train_end = total - test_window_months

        if max_train_end <= min_train_months:
            logger.warning(
                f"[expanding] データ期間が短すぎてフォールドを生成できません "
                f"（全期間: {total}ヶ月, 最小学習期間: {min_train_months}ヶ月, "
                f"テストウィンドウ: {test_window_months}ヶ月）"
            )
            return []

        available = max_train_end - min_train_months
        step = max(1, available // n_splits)

        folds = []
        for i in range(n_splits):
            train_end = min_train_months + i * step
            if train_end > max_train_end:
                break
            test_end = min(train_end + test_window_months, total)
            train_periods = all_periods[:train_end]
            test_periods = all_periods[train_end:test_end]
            if len(test_periods) == 0:
                break
            folds.append((train_periods, test_periods))

        return folds

    def _generate_rolling_folds(
        self,
        all_periods: pd.PeriodIndex,
        n_splits: int,
        train_window_months: int,
        test_window_months: int
    ) -> List[Tuple]:
        """
        固定ウィンドウ（ローリング）フォールドを生成する。
        学習データ幅を固定したまま、train/test ウィンドウをスライドさせる。

        Args:
            all_periods: 月単位のユニーク期間（昇順）
            n_splits: フォールド数
            train_window_months: 固定学習幅（月数）
            test_window_months: テストウィンドウ幅（月数）

        Returns:
            List of (train_periods, test_periods) tuples
        """
        total = len(all_periods)
        min_required = train_window_months + test_window_months

        if total < min_required:
            logger.warning(
                f"[rolling] データ期間が短すぎてフォールドを生成できません "
                f"（全期間: {total}ヶ月, 学習幅: {train_window_months}ヶ月, "
                f"テストウィンドウ: {test_window_months}ヶ月）"
            )
            return []

        max_start = total - train_window_months - test_window_months
        step = max(1, max_start // n_splits)

        folds = []
        for i in range(n_splits):
            train_start = i * step
            train_end = train_start + train_window_months
            test_end = min(train_end + test_window_months, total)
            if train_end >= total:
                break
            train_periods = all_periods[train_start:train_end]
            test_periods = all_periods[train_end:test_end]
            if len(test_periods) == 0:
                break
            folds.append((train_periods, test_periods))

        return folds

    def _cv_valid_months(self) -> int:
        """early stopping / キャリブレーション用ホールドアウトの月数を返す。

        config['model']['cross_validation']['valid_window_months'] で指定可能。
        既定は 2 ヶ月。
        """
        cv_cfg = self.config.get('model', {}).get('cross_validation', {})
        return int(cv_cfg.get('valid_window_months', 2))

    @staticmethod
    def _split_train_valid_periods(
        train_periods: pd.PeriodIndex, valid_months: int
    ) -> Tuple[pd.PeriodIndex, pd.PeriodIndex]:
        """学習期間を fit 用と early-stopping/calibration 用に時系列分割する。

        train_periods の末尾 valid_months ヶ月を検証（valid）ホールドアウトとし、
        残りを fit 期間とする。test に隣接する直近期間を valid に充てることで
        時系列順序を保つ。

        train_periods が短く valid を確保できない場合（fit 側が空になる場合）は
        分割せず、(train_periods, 空) を返す。呼び出し側は valid が空なら
        従来どおり test を early stopping に使うフォールバックを行う。

        Args:
            train_periods: フォールドの学習期間（昇順の月次 PeriodIndex）
            valid_months: valid ホールドアウトの月数

        Returns:
            (fit_periods, valid_periods)
        """
        n = len(train_periods)
        vm = max(1, valid_months)
        # fit 側に最低1ヶ月残す。確保できないなら分割しない。
        if n - vm < 1:
            return train_periods, train_periods[:0]
        return train_periods[:-vm], train_periods[-vm:]

    # ------------------------------------------------------------------
    # クロスバリデーション
    # ------------------------------------------------------------------

    def cross_validate(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        n_splits: int = 5,
        min_train_months: int = 12,
        train_window_months: int = 12,
        test_window_months: int = 3,
        cv_type: str = 'expanding',
        evaluator: Optional[ModelEvaluator] = None,
        bet_type: str = 'win',
        threshold: float = 1.05,
    ) -> Dict:
        """
        日付ベースの時系列交差検証。

        Args:
            df: データフレーム（'date' カラム必須）
            feature_cols: 特徴量カラム
            target_col: ターゲットカラム（'target_win' など）
            n_splits: フォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            train_window_months: 固定窓の学習幅（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding'（拡張窓）or 'rolling'（固定窓）
            evaluator: ModelEvaluator インスタンス（回収率計算用、省略可）
            bet_type: 賭けの種類（'win' / 'place'）
            threshold: 回収率計算に使う最低期待値（EV = 予測確率 × オッズ）

        Returns:
            Dict: fold_results リストと集計統計
        """
        logger.info(
            f"時系列CV開始（type={cv_type}, {n_splits}分割, "
            f"test_window={test_window_months}ヶ月）"
        )

        df = df.sort_values('date').reset_index(drop=True)

        # 月次 period はフォールドごとに再計算せず1回だけ計算する
        month_period = df['date'].dt.to_period('M')
        all_periods = pd.PeriodIndex(month_period.unique()).sort_values()
        if cv_type == 'rolling':
            folds = self._generate_rolling_folds(
                all_periods, n_splits, train_window_months, test_window_months
            )
        else:
            folds = self._generate_cv_folds(
                all_periods, n_splits, min_train_months, test_window_months
            )

        empty_result = {
            'fold_results': [],
            'mean_auc': 0.0, 'std_auc': 0.0,
            'mean_hit_rate': 0.0, 'std_hit_rate': 0.0,
            'mean_recovery_rate': 0.0, 'std_recovery_rate': 0.0,
        }

        if not folds:
            logger.warning("有効なフォールドが生成されませんでした。CVをスキップします。")
            return empty_result

        fold_results = []
        for fold_idx, (train_periods, test_periods) in enumerate(folds):
            fold_num = fold_idx + 1
            train_start = str(train_periods[0])
            train_end = str(train_periods[-1])
            test_start = str(test_periods[0])
            test_end = str(test_periods[-1])

            logger.info(
                f"Fold {fold_num}/{len(folds)}: "
                f"Train [{train_start}〜{train_end}] → Test [{test_start}〜{test_end}]"
            )

            test_mask = month_period.isin(test_periods)
            test_df = df[test_mask]

            # early stopping 検証セットを test から分離する（楽観バイアス防止）。
            # train_periods の末尾を valid ホールドアウトに切り出し、fit には残りを使う。
            # valid を確保できない場合のみ、従来どおり test を early stopping に使う。
            fit_periods, valid_periods = self._split_train_valid_periods(
                train_periods, self._cv_valid_months()
            )
            fit_mask = month_period.isin(fit_periods)
            fit_df = df[fit_mask]

            if len(fit_df) == 0 or len(test_df) == 0:
                logger.warning(f"Fold {fold_num}: データが空のためスキップ")
                continue

            if len(valid_periods) > 0:
                valid_df = df[month_period.isin(valid_periods)]
                X_val_es = valid_df[feature_cols]
                y_val_es = valid_df[target_col]
            else:
                # フォールバック: train が短く valid を取れないため test を使う
                X_val_es = test_df[feature_cols]
                y_val_es = test_df[target_col]

            X_train = fit_df[feature_cols]
            y_train = fit_df[target_col]
            X_test = test_df[feature_cols]
            y_test = test_df[target_col]

            cv_model_name = f'_cv_{target_col}_fold{fold_num}'
            decay_cfg = self.config.get('model', {}).get('time_decay', {})
            cv_sample_weight = self._trainer.compute_time_decay_weights(
                fit_df['date'],
                recent_years=float(decay_cfg.get('recent_years', 2.0)),
                mid_years=float(decay_cfg.get('mid_years', 4.0)),
                w_recent=float(decay_cfg.get('w_recent', 3.0)),
                w_mid=float(decay_cfg.get('w_mid', 2.0)),
            )
            model = self._trainer.train(
                X_train, y_train, X_val_es, y_val_es,
                model_name=cv_model_name,
                sample_weight=cv_sample_weight,
            )

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            auc = ModelEvaluator.compute_auc(y_test, y_pred)

            hit_rate = float('nan')
            recovery_rate = float('nan')
            if (evaluator is not None
                    and 'finish_position' in test_df.columns
                    and has_two_source_odds(test_df, bet_type)):
                rr_result = evaluator.calculate_recovery_rate(
                    test_df, y_pred, min_ev=threshold, bet_type=bet_type
                )
                hit_rate = rr_result.get('hit_rate', float('nan'))
                recovery_rate = rr_result.get('recovery_rate', float('nan'))

            result = {
                'fold': fold_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(fit_df),
                'test_size': len(test_df),
                'auc': auc,
                'hit_rate': hit_rate,
                'recovery_rate': recovery_rate,
            }
            fold_results.append(result)

            msg = f"Fold {fold_num} 結果: AUC={auc:.4f}"
            if not np.isnan(hit_rate):
                msg += f" | 的中率={hit_rate:.1f}%"
            if not np.isnan(recovery_rate):
                msg += f" | 回収率={recovery_rate:.1f}%"
            logger.info(msg)

            self._trainer.models.pop(cv_model_name, None)

        if not fold_results:
            return empty_result

        aucs = [r['auc'] for r in fold_results if not np.isnan(r['auc'])]
        hit_rates = [r['hit_rate'] for r in fold_results if not np.isnan(r['hit_rate'])]
        recovery_rates = [r['recovery_rate'] for r in fold_results if not np.isnan(r['recovery_rate'])]

        summary = {
            'fold_results': fold_results,
            'mean_auc': float(np.mean(aucs)) if aucs else 0.0,
            'std_auc': float(np.std(aucs)) if aucs else 0.0,
            'mean_hit_rate': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'std_hit_rate': float(np.std(hit_rates)) if hit_rates else 0.0,
            'mean_recovery_rate': float(np.mean(recovery_rates)) if recovery_rates else 0.0,
            'std_recovery_rate': float(np.std(recovery_rates)) if recovery_rates else 0.0,
        }

        msg = (
            f"CV完了 - AUC: {summary['mean_auc']:.4f} (+/-{summary['std_auc']:.4f})"
        )
        if hit_rates:
            msg += f" | 的中率: {summary['mean_hit_rate']:.1f}% (+/-{summary['std_hit_rate']:.1f}%)"
        if recovery_rates:
            msg += f" | 回収率: {summary['mean_recovery_rate']:.1f}% (+/-{summary['std_recovery_rate']:.1f}%)"
        logger.info(msg)

        return summary

    def cross_validate_ranking(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ranking_label_config: dict,
        n_splits: int = 5,
        min_train_months: int = 12,
        train_window_months: int = 12,
        test_window_months: int = 3,
        cv_type: str = 'expanding',
        evaluator: Optional[ModelEvaluator] = None,
    ) -> Dict:
        """
        ランキングモデルの時系列交差検証。

        バイナリCV（cross_validate）との主な違い:
        - 評価指標は AUC ではなく top-1 的中率（予測1位馬が実際1着か）
        - lgb.Dataset に group パラメータが必要なため、fold内でグループを再構築する

        Args:
            df: データフレーム（'date', 'race_id', 'finish_position' カラム必須）
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label']
            n_splits: フォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            train_window_months: 固定窓の学習幅（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding' or 'rolling'
            evaluator: ModelEvaluator インスタンス（回収率計算用、省略可）

        Returns:
            Dict: fold_results リストと集計統計
        """
        logger.info(
            f"ランキング時系列CV開始（type={cv_type}, {n_splits}分割, "
            f"test_window={test_window_months}ヶ月）"
        )

        if 'finish_position' not in df.columns:
            logger.error("finish_position カラムがないためランキングCVをスキップします")
            return {'fold_results': [], 'mean_hit_rate': 0.0, 'std_hit_rate': 0.0,
                    'mean_recovery_rate': 0.0, 'std_recovery_rate': 0.0}

        df = df.sort_values('date').reset_index(drop=True)
        month_period = df['date'].dt.to_period('M')  # 1回だけ計算
        all_periods = pd.PeriodIndex(month_period.unique()).sort_values()

        if cv_type == 'rolling':
            folds = self._generate_rolling_folds(
                all_periods, n_splits, train_window_months, test_window_months
            )
        else:
            folds = self._generate_cv_folds(
                all_periods, n_splits, min_train_months, test_window_months
            )

        empty_result = {
            'fold_results': [],
            'mean_hit_rate': 0.0, 'std_hit_rate': 0.0,
            'mean_recovery_rate': 0.0, 'std_recovery_rate': 0.0,
        }

        if not folds:
            logger.warning("有効なフォールドが生成されませんでした。ランキングCVをスキップします。")
            return empty_result

        fold_results = []
        for fold_idx, (train_periods, test_periods) in enumerate(folds):
            fold_num = fold_idx + 1
            train_start = str(train_periods[0])
            train_end = str(train_periods[-1])
            test_start = str(test_periods[0])
            test_end = str(test_periods[-1])

            logger.info(
                f"ランキングCV Fold {fold_num}/{len(folds)}: "
                f"Train [{train_start}〜{train_end}] → Test [{test_start}〜{test_end}]"
            )

            test_mask = month_period.isin(test_periods)
            test_df = df[test_mask].sort_values(['date', 'race_id']).reset_index(drop=True)

            # early stopping 検証セットを test から分離する（楽観バイアス防止）。
            fit_periods, valid_periods = self._split_train_valid_periods(
                train_periods, self._cv_valid_months()
            )
            fit_df = (
                df[month_period.isin(fit_periods)]
                .sort_values(['date', 'race_id']).reset_index(drop=True)
            )

            if len(fit_df) == 0 or len(test_df) == 0:
                logger.warning(f"ランキングCV Fold {fold_num}: データが空のためスキップ")
                continue

            # ラベル生成は make_relevance_labels に集約し、本学習
            # （prepare_ranking_data）と定義を一致させる。
            # odds_weighted_dynamic の閾値は fit_df のみで計算しリークを防ぐ。
            scheme = ranking_label_config.get('scheme', 'positional')
            dynamic_thr = None
            if scheme == 'odds_weighted_dynamic' and 'odds_win' in df.columns:
                dynamic_thr = compute_dynamic_odds_thresholds(fit_df)
            y_train = pd.Series(
                make_relevance_labels(fit_df, ranking_label_config, dynamic_thr),
                index=fit_df.index,
            )
            y_test = pd.Series(
                make_relevance_labels(test_df, ranking_label_config, dynamic_thr),
                index=test_df.index,
            )

            groups_train = fit_df.groupby('race_id', sort=False).size().tolist()
            X_train = fit_df[feature_cols]
            X_test = test_df[feature_cols]

            # valid ホールドアウト（early stopping 用）を用意。取れない場合は test を使う。
            if len(valid_periods) > 0:
                valid_df = (
                    df[month_period.isin(valid_periods)]
                    .sort_values(['date', 'race_id']).reset_index(drop=True)
                )
                X_val_es = valid_df[feature_cols]
                y_val_es = pd.Series(
                    make_relevance_labels(valid_df, ranking_label_config, dynamic_thr),
                    index=valid_df.index,
                )
                groups_val_es = valid_df.groupby('race_id', sort=False).size().tolist()
            else:
                X_val_es = X_test
                y_val_es = y_test
                groups_val_es = test_df.groupby('race_id', sort=False).size().tolist()

            cv_model_name = f'_cv_ranking_fold{fold_num}'
            self._trainer.train_ranking(
                X_train, y_train, groups_train,
                X_val_es, y_val_es, groups_val_es,
                model_name=cv_model_name,
            )
            model = self._trainer.models[cv_model_name]

            scores = model.predict(X_test, num_iteration=model.best_iteration)
            test_eval_df = test_df.copy()
            test_eval_df['_score'] = scores

            hit_count = 0
            race_count = 0
            for _, race in test_eval_df.groupby('race_id'):
                if race.empty:
                    continue
                top_horse = race.loc[race['_score'].idxmax()]
                if int(top_horse['finish_position']) == 1:
                    hit_count += 1
                race_count += 1
            top1_hit_rate = hit_count / race_count * 100 if race_count > 0 else float('nan')

            recovery_rate = float('nan')
            if (evaluator is not None
                    and has_two_source_odds(test_df, 'win')
                    and race_count > 0):
                try:
                    temperature = self.config.get('evaluation', {}).get('ranking_temperature', 1.5)
                    min_ev = self.config.get('evaluation', {}).get('min_ev_win', 1.05)
                    recovery_rate = evaluator.calculate_ranking_recovery_rate(
                        test_eval_df, temperature, min_ev
                    )
                except Exception as e:
                    logger.debug(f"ランキングCV 回収率計算エラー（スキップ）: {e}")

            result = {
                'fold': fold_num,
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'train_size': len(fit_df),
                'test_size': len(test_df),
                'hit_rate': top1_hit_rate,
                'recovery_rate': recovery_rate,
            }
            fold_results.append(result)

            msg = f"ランキングCV Fold {fold_num} 結果: top1的中率={top1_hit_rate:.1f}%"
            if not np.isnan(recovery_rate):
                msg += f" | 回収率={recovery_rate:.1f}%"
            logger.info(msg)

            self._trainer.models.pop(cv_model_name, None)

        if not fold_results:
            return empty_result

        hit_rates = [r['hit_rate'] for r in fold_results if not np.isnan(r['hit_rate'])]
        recovery_rates = [r['recovery_rate'] for r in fold_results if not np.isnan(r['recovery_rate'])]

        summary = {
            'fold_results': fold_results,
            'mean_hit_rate': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'std_hit_rate': float(np.std(hit_rates)) if hit_rates else 0.0,
            'mean_recovery_rate': float(np.mean(recovery_rates)) if recovery_rates else 0.0,
            'std_recovery_rate': float(np.std(recovery_rates)) if recovery_rates else 0.0,
        }

        msg = (
            f"ランキングCV完了 - top1的中率: {summary['mean_hit_rate']:.1f}%"
            f" (+/-{summary['std_hit_rate']:.1f}%)"
        )
        if recovery_rates:
            msg += (
                f" | 回収率: {summary['mean_recovery_rate']:.1f}%"
                f" (+/-{summary['std_recovery_rate']:.1f}%)"
            )
        logger.info(msg)

        return summary

    # ------------------------------------------------------------------
    # ブレンド係数最適化
    # ------------------------------------------------------------------

    def optimize_blend_alpha_cv(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ranking_label_config: dict,
        min_ev_win: float = 1.05,
        min_ev_place: float = 1.05,
        n_splits: int = 3,
        min_train_months: int = 12,
        test_window_months: int = 3,
        cv_type: str = 'expanding',
        n_trials: int = 30,
    ) -> Dict:
        """
        Optunaを使いCVブレンド係数 (blend_alpha, blend_alpha_place, ranking_temperature) を最適化する。

        各CVフォールドで binary win/place モデルと ranking モデルを学習して予測を収集し、
        Optuna がその予測上でアルファと温度パラメータを最大化（単勝・複勝CV回収率の平均）する。

        Args:
            df: 特徴量DataFrame（'date', 'race_id', 'target_win', 'finish_position',
                'odds_win' カラム必須。'target_place' と複勝の締切前オッズ
                （odds_pre_place）・確定払戻（payout_place）が揃っていれば複勝も最適化）
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label'] の辞書
            min_ev_win: 単勝購入の最低期待値
            min_ev_place: 複勝購入の最低期待値
            n_splits: CVフォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding' or 'rolling'
            n_trials: Optuna 試行回数

        Returns:
            Dict: best_alpha, best_alpha_place, best_temperature, best_recovery_rate。
                  失敗時は空辞書。
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optunaがインストールされていません。blend_alpha最適化をスキップします。")
            return {}

        required_cols = {
            BET_ODDS_COLS['win'], PAYOUT_COLS['win'],
            'finish_position', 'target_win', 'race_id',
        }
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.warning(f"必須カラム {missing} がないためblend最適化をスキップします。")
            return {}

        # 複勝も2系統オッズ（締切前オッズ＋確定払戻）が揃っている場合のみ最適化する
        has_place_data = 'target_place' in df.columns and has_two_source_odds(df, 'place')
        if has_place_data:
            logger.info(
                "'target_place' と複勝の締切前オッズ・確定払戻が存在するため "
                "blend_alpha_place も最適化します。"
            )
        else:
            logger.info("'target_place' が存在しないため blend_alpha_place の最適化をスキップします。")

        logger.info("=== blend_alpha / blend_alpha_place / ranking_temperature Optuna最適化開始 ===")
        df = df.sort_values('date').reset_index(drop=True)
        month_period = df['date'].dt.to_period('M')  # 1回だけ計算
        all_periods = pd.PeriodIndex(month_period.unique()).sort_values()

        if cv_type == 'rolling':
            folds = self._generate_rolling_folds(
                all_periods, n_splits, 12, test_window_months
            )
        else:
            folds = self._generate_cv_folds(
                all_periods, n_splits, min_train_months, test_window_months
            )

        if not folds:
            logger.warning("有効なフォールドが生成されませんでした。blend最適化をスキップします。")
            return {}

        scheme = ranking_label_config.get('scheme', 'positional')

        fold_data: List[Dict] = []
        for fold_idx, (train_periods, test_periods) in enumerate(folds):
            fold_num = fold_idx + 1
            logger.info(f"blend最適化 Fold {fold_num}/{len(folds)}: モデル学習中...")

            test_mask = month_period.isin(test_periods)
            test_df_base = df[test_mask].sort_values(['date', 'race_id']).reset_index(drop=True)

            # early stopping / キャリブレーションを test から分離する（楽観バイアス防止）。
            # train_periods の末尾を valid ホールドアウトに切り出し、fit には残りを使う。
            fit_periods, valid_periods = self._split_train_valid_periods(
                train_periods, self._cv_valid_months()
            )
            fit_df = (
                df[month_period.isin(fit_periods)]
                .sort_values(['date', 'race_id']).reset_index(drop=True)
            )

            if len(fit_df) == 0 or len(test_df_base) == 0:
                continue
            if fit_df['target_win'].sum() == 0:
                continue

            # valid ホールドアウト。取れない場合のみ test をフォールバックに使う。
            use_test_as_valid = len(valid_periods) == 0
            if use_test_as_valid:
                logger.warning(
                    f"blend最適化 Fold {fold_num}: 学習期間が短く valid ホールドアウトを"
                    f"確保できないため、early stopping/calibration に test を使います"
                    f"（回収率が楽観的になる可能性があります）"
                )
                valid_df = test_df_base
            else:
                valid_df = (
                    df[month_period.isin(valid_periods)]
                    .sort_values(['date', 'race_id']).reset_index(drop=True)
                )

            X_train_bin = fit_df[list(feature_cols)]
            y_train_bin = fit_df['target_win']
            X_val_bin = valid_df[list(feature_cols)]
            y_val_bin = valid_df['target_win']
            X_test_bin = test_df_base[list(feature_cols)]

            bin_name = f'_blend_opt_win_fold{fold_num}'
            # config['model']['time_decay'] が正しいパス。以前は
            # self.config.get('time_decay') でルート直下を見ており、
            # 常にデフォルトウェイトで学習していた（本学習と不整合）。
            decay_cfg = self.config.get('model', {}).get('time_decay', {})
            sw = self._trainer.compute_time_decay_weights(
                fit_df['date'],
                recent_years=float(decay_cfg.get('recent_years', 2.0)),
                mid_years=float(decay_cfg.get('mid_years', 4.0)),
                w_recent=float(decay_cfg.get('w_recent', 3.0)),
                w_mid=float(decay_cfg.get('w_mid', 2.0)),
            )
            # early stopping は valid で行い、calibration も valid で fit する
            # （test は blend 評価用の予測にのみ使う）。
            self._trainer.train(X_train_bin, y_train_bin, X_val_bin, y_val_bin,
                                model_name=bin_name, sample_weight=sw)
            if y_val_bin.sum() > 0:
                self._trainer.calibrate_model(bin_name, X_val_bin, y_val_bin, method='isotonic')
            win_proba: np.ndarray = self._trainer.predict(X_test_bin, model_name=bin_name)

            place_proba: np.ndarray | None = None
            odds_place_arr: np.ndarray | None = None
            payout_place_arr: np.ndarray | None = None
            target_place_arr: np.ndarray | None = None
            if has_place_data and fit_df['target_place'].sum() > 0:
                place_name = f'_blend_opt_place_fold{fold_num}'
                y_train_place = fit_df['target_place']
                y_val_place = valid_df['target_place']
                self._trainer.train(X_train_bin, y_train_place, X_val_bin, y_val_place,
                                    model_name=place_name, sample_weight=sw)
                if y_val_place.sum() > 0:
                    self._trainer.calibrate_model(place_name, X_val_bin, y_val_place, method='isotonic')
                place_proba = self._trainer.predict(X_test_bin, model_name=place_name)
                # 賭け判断＝締切前複勝オッズ、払戻＝確定複勝払戻（2系統に分離）
                odds_place_arr = test_df_base[BET_ODDS_COLS['place']].values.astype(float)
                payout_place_arr = (
                    test_df_base[PAYOUT_COLS['place']].fillna(0.0).values.astype(float) / 100.0
                )
                target_place_arr = test_df_base['target_place'].values
                self._trainer.models.pop(place_name, None)
                self._trainer.calibrators.pop(place_name, None)

            fit_df_r = fit_df
            valid_df_r = valid_df
            test_df_r = test_df_base.copy()
            ranking_fcols = list(feature_cols)

            # ラベル生成は本学習・CV と共通の make_relevance_labels を使う。
            # odds_weighted_dynamic の閾値は fit_df のみで計算しリークを防ぐ。
            dynamic_thr = None
            if scheme == 'odds_weighted_dynamic' and 'odds_win' in fit_df_r.columns:
                dynamic_thr = compute_dynamic_odds_thresholds(fit_df_r)
            y_train_r = pd.Series(
                make_relevance_labels(fit_df_r, ranking_label_config, dynamic_thr),
                index=fit_df_r.index,
            )
            y_val_r = pd.Series(
                make_relevance_labels(valid_df_r, ranking_label_config, dynamic_thr),
                index=valid_df_r.index,
            )
            groups_train = fit_df_r.groupby('race_id', sort=False).size().tolist()
            groups_val = valid_df_r.groupby('race_id', sort=False).size().tolist()

            rank_name = f'_blend_opt_rank_fold{fold_num}'
            self._trainer.train_ranking(
                fit_df_r[ranking_fcols], y_train_r, groups_train,
                valid_df_r[ranking_fcols], y_val_r, groups_val,
                model_name=rank_name,
            )
            ranking_scores: np.ndarray = self._trainer.predict(
                test_df_r[ranking_fcols], model_name=rank_name
            )

            # 賭け判断＝締切前オッズ、払戻＝確定払戻金（2系統に分離）
            _bet_odds_win = test_df_r[BET_ODDS_COLS['win']].values.astype(float)
            fold_entry: Dict = {
                'win_proba': win_proba,
                'ranking_scores': ranking_scores,
                'race_id': test_df_r['race_id'].values,
                'odds_win': _bet_odds_win,
                'payout_win': (
                    test_df_r[PAYOUT_COLS['win']].fillna(0.0).values.astype(float) / 100.0
                ),
                # 締切前オッズ欠損行は賭け対象から外す（確定オッズで代用しない）
                'valid_win': np.isfinite(_bet_odds_win) & (_bet_odds_win > 0),
                'finish_position': test_df_r['finish_position'].values,
            }
            # objective のベクトル化用に「1着マスク」を事前計算する
            _fin = test_df_r['finish_position'].to_numpy()
            fold_entry['won1'] = (~pd.isna(_fin)) & (np.nan_to_num(_fin, nan=-1).astype(int) == 1)
            if place_proba is not None:
                fold_entry['place_proba'] = place_proba
                fold_entry['odds_place'] = odds_place_arr
                fold_entry['payout_place'] = payout_place_arr
                fold_entry['target_place'] = target_place_arr
                fold_entry['valid_place'] = (
                    np.isfinite(odds_place_arr) & (odds_place_arr > 0)
                )
            fold_data.append(fold_entry)

            self._trainer.models.pop(bin_name, None)
            self._trainer.calibrators.pop(bin_name, None)
            self._trainer.models.pop(rank_name, None)

        if not fold_data:
            logger.warning("有効なフォールドデータがありません。blend最適化をスキップします。")
            return {}

        logger.info(
            f"フォールドデータ収集完了（{len(fold_data)}フォールド）。"
            f"Optuna最適化開始（{n_trials}試行）..."
        )

        from src.predict.blend_ensemble import ranking_to_proba_per_race

        optimize_place = has_place_data and any('place_proba' in fd for fd in fold_data)

        def objective(trial: 'optuna.Trial') -> float:  # type: ignore[name-defined]
            alpha = trial.suggest_float('blend_alpha', 0.3, 0.9)
            temperature = trial.suggest_float('ranking_temperature', 0.5, 3.0)
            alpha_place = (
                trial.suggest_float('blend_alpha_place', 0.05, 0.5)
                if optimize_place else alpha
            )
            win_rr_list: List[float] = []
            place_rr_list: List[float] = []
            for fd in fold_data:
                # レース単位で softmax を取る。フォールド全体に ranking_to_proba を
                # 直接適用すると数千レースを跨いで正規化され、rankingの寄与が
                # 1/N ≈ 0 に潰れて blend_alpha が「rankingを無視する値」に最適化される。
                ranking_proba = ranking_to_proba_per_race(
                    fd['ranking_scores'], fd['race_id'], temperature
                )
                blended_win = alpha * fd['win_proba'] + (1.0 - alpha) * ranking_proba
                # 賭け判断は締切前オッズ、払戻は確定払戻金
                ev_win = blended_win * np.nan_to_num(fd['odds_win'])
                bet_mask_win = fd['valid_win'] & (ev_win >= min_ev_win)
                if not bet_mask_win.any():
                    win_rr_list.append(0.0)
                else:
                    # ジェネレータ和を廃止しベクトル集計
                    total_bet = float(bet_mask_win.sum()) * 100.0
                    total_return = 100.0 * fd['payout_win'][bet_mask_win & fd['won1']].sum()
                    win_rr_list.append((total_return / total_bet * 100.0) if total_bet > 0 else 0.0)
                if optimize_place and 'place_proba' in fd:
                    blended_place = alpha_place * fd['place_proba'] + (1.0 - alpha_place) * ranking_proba
                    ev_place = blended_place * np.nan_to_num(fd['odds_place'])
                    bet_mask_place = fd['valid_place'] & (ev_place >= min_ev_place)
                    if not bet_mask_place.any():
                        place_rr_list.append(0.0)
                    else:
                        place_won = fd['target_place'] == 1
                        total_bet_p = float(bet_mask_place.sum()) * 100.0
                        total_return_p = (
                            100.0 * fd['payout_place'][bet_mask_place & place_won].sum()
                        )
                        place_rr_list.append(
                            (total_return_p / total_bet_p * 100.0) if total_bet_p > 0 else 0.0
                        )

            win_mean = float(np.mean(win_rr_list)) if win_rr_list else 0.0
            if place_rr_list:
                place_mean = float(np.mean(place_rr_list))
                return (win_mean + place_mean) / 2.0
            return win_mean

        from optuna.samplers import TPESampler

        sampler = TPESampler(n_startup_trials=20, multivariate=True, seed=42)
        # trial.report() を呼ばない設計のため MedianPruner は機能しない。設定しない。
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_alpha: float = study.best_params['blend_alpha']
        best_temperature: float = study.best_params['ranking_temperature']
        best_alpha_place: float = (
            study.best_params['blend_alpha_place'] if optimize_place else best_alpha
        )
        best_rr: float = study.best_value

        logger.info(
            f"blend最適化完了: best_alpha={best_alpha:.3f}, "
            f"best_alpha_place={best_alpha_place:.3f}, "
            f"best_temperature={best_temperature:.3f}, "
            f"CV回収率={best_rr:.1f}%"
        )
        return {
            'best_alpha': best_alpha,
            'best_alpha_place': best_alpha_place,
            'best_temperature': best_temperature,
            'best_recovery_rate': best_rr,
        }

    # ------------------------------------------------------------------
    # ハイパーパラメータ最適化
    # ------------------------------------------------------------------

    def hyperparameter_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 100,
        model_name: str = 'win',
        val_meta_df: Optional[pd.DataFrame] = None,
        objective_metric: str = 'auc',
        fobj_sample_weight: Optional[np.ndarray] = None,
        fobj_beta_range: Optional[Tuple[float, float]] = None,
    ) -> Dict:
        """
        Optunaを使用したハイパーパラメータチューニング

        Args:
            X_train: 学習データ
            y_train: 学習ラベル
            X_val: 検証データ
            y_val: 検証ラベル
            n_trials: 試行回数
            model_name: モデル識別子（'win' or 'place'）。
                'place' のときは config の lightgbm_place セクションを
                ベースパラメータとして使用する。
            val_meta_df: 回収率計算用の検証メタデータ。
                賭け判断用の締切前オッズ（odds_pre_*）と払戻用の確定払戻金（payout_*）、
                および的中判定列（finish_position / target_umaren）が必須。
                objective_metric='recovery_rate' の場合に使用。
            objective_metric: 最適化指標。'auc'（デフォルト）または 'recovery_rate'。
                'recovery_rate' は賭け判断＝締切前オッズ・払戻＝確定払戻金と
                オッズ源を分離しているためリークしない。
            fobj_sample_weight: Phase 2b 用。オッズ重み（``compute_odds_weights`` の戻り値。
                時間減衰との合成済みでよい）。``fobj_beta_range`` と併せて指定した場合のみ、
                各 trial で custom fobj（``beta`` を探索）を使って学習する。
                指定が無ければ従来通り標準の binary objective を使う。
            fobj_beta_range: Phase 2b 用。``beta`` の探索範囲 ``(low, high)``。
                ``fobj_sample_weight`` と両方指定した場合のみ有効。
                ``objective_metric='recovery_rate'`` と併用することを想定
                （AUC で beta を最適化しても回収率への寄与は測れないため）。

        Returns:
            Dict: 最適なパラメータ（チューナブルキーのみ。fobj 探索時は ``fobj_beta`` を含む）
        """
        logger.info(f"[{model_name}] ハイパーパラメータチューニング開始")

        try:
            import optuna
        except ImportError:
            logger.error("Optunaがインストールされていません")
            return {}

        # モデルごとの固定パラメータをベースとして使用
        model_config = self._trainer.config
        if model_name == 'place':
            base_fixed = model_config.get('lightgbm_place', model_config.get('lightgbm', {}))
        else:
            base_fixed = model_config.get('lightgbm', {})

        # チューナブルキー以外（objective / metric / is_unbalance 等）は固定値として引き継ぐ
        _TUNABLE_KEYS = frozenset({
            'num_leaves', 'learning_rate', 'feature_fraction',
            'bagging_fraction', 'bagging_freq', 'min_data_in_leaf',
            'lambda_l1', 'lambda_l2',
        })
        fixed_params = {k: v for k, v in base_fixed.items() if k not in _TUNABLE_KEYS}
        # LightGBM が受け付けないキーを除去
        for drop_key in ('n_estimators', 'early_stopping_rounds'):
            fixed_params.pop(drop_key, None)
        fixed_params['verbosity'] = -1

        # Dataset は試行ごとに再構築せず1回だけ構築して再利用する
        # （ヒストグラムビニングの再計算を回避）。
        # min_data_in_leaf 等をtrialで変えるため feature_pre_filter は無効化必須。
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = train_data.create_valid(X_val, label=y_val)

        # 回収率目的の事前計算（min_ev 非依存の量は objective 外で1回だけ）。
        # 賭け判断は締切前オッズ、払戻は確定払戻金と分離するのでリークしない。
        rr_odds = rr_payout = rr_hit = rr_min_ev = None
        if objective_metric == 'recovery_rate':
            if val_meta_df is None:
                raise ValueError(
                    "objective_metric='recovery_rate' には val_meta_df が必要です"
                    "（締切前オッズ・確定払戻列を含む検証メタデータ）"
                )
            bet_type = model_name if model_name in BET_ODDS_COLS else 'win'
            bet_col, payout_col = BET_ODDS_COLS[bet_type], PAYOUT_COLS[bet_type]
            missing = [c for c in (bet_col, payout_col) if c not in val_meta_df.columns]
            if missing:
                raise ValueError(
                    f"objective_metric='recovery_rate' に必要な列がありません: {missing}"
                )
            rr_odds = val_meta_df[bet_col].to_numpy(dtype=float)
            # 払戻金は100円あたりなので倍率へ換算
            rr_payout = val_meta_df[payout_col].fillna(0.0).to_numpy(dtype=float) / 100.0
            if bet_type == 'umaren':
                rr_hit = val_meta_df['target_umaren'].to_numpy() == 1
            else:
                rr_finish = val_meta_df['finish_position'].to_numpy()
                finish_int = np.nan_to_num(rr_finish, nan=-1).astype(int)
                rr_hit = (~pd.isna(rr_finish)) & (
                    finish_int <= 3 if bet_type == 'place' else finish_int == 1
                )
            # 締切前オッズ欠損行は評価対象外（確定オッズで代用しない）
            rr_valid = np.isfinite(rr_odds) & (rr_odds > 0)
            n_dropped = int((~rr_valid).sum())
            if n_dropped > 0:
                logger.warning(
                    f"[{model_name}] HPO 回収率: 締切前オッズ欠損の {n_dropped}/{len(rr_odds)} 行を"
                    "評価対象外にしました"
                )
            if not rr_valid.any():
                raise ValueError(
                    f"[{model_name}] HPO 回収率: 締切前オッズが有効な行が0件です"
                )
            rr_min_ev = self.config.get('evaluation', {}).get(
                f'min_ev_{bet_type}', 1.05
            )

        # Phase 2b: custom fobj の beta を trial ごとに探索する場合の事前計算。
        # fixed_params の scale_pos_weight/is_unbalance は fobj 使用時 LightGBM に無視されるため、
        # pos_weight として fobj に明示的に渡す（train() 本体と同じ計算式で揃える）。
        use_fobj_search = fobj_sample_weight is not None and fobj_beta_range is not None
        fobj_pos_weight = 1.0
        if use_fobj_search:
            fobj_pos_weight = LightGBMTrainer.compute_auto_pos_weight(
                model_name, y_train, fixed_params
            )
            fixed_params = {
                k: v for k, v in fixed_params.items()
                if k not in ('scale_pos_weight', 'is_unbalance')
            }
            logger.info(
                f"[{model_name}] HPO で custom fobj の beta を探索します: "
                f"range={fobj_beta_range} pos_weight={fobj_pos_weight:.2f}"
            )

        def objective(trial):
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
            params = {
                **fixed_params,
                'feature_pre_filter': False,
                'boosting_type': boosting_type,
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 15.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 15.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
            }
            if boosting_type == 'dart':
                params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.3)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

            if use_fobj_search:
                beta = trial.suggest_float('fobj_beta', *fobj_beta_range)
                params['objective'] = LightGBMTrainer.make_odds_weighted_fobj(
                    fobj_sample_weight, y_train.to_numpy(), beta=beta, pos_weight=fobj_pos_weight,
                )
                params['metric'] = 'auc'

            # DART は early stopping が機能しないため n_estimators を探索する
            if boosting_type == 'dart':
                num_boost_round = trial.suggest_int('n_estimators', 200, 1000)
                callbacks_list = [lgb.log_evaluation(period=0)]
            else:
                num_boost_round = 1000
                callbacks_list = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                callbacks=callbacks_list,
            )

            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            if use_fobj_search:
                # custom fobj 使用時 predict() は常にロジットを返すため確率に変換する
                y_pred = 1.0 / (1.0 + np.exp(-y_pred))

            if rr_odds is not None:
                # 賭け判断＝締切前オッズ、払戻＝確定払戻金（2系統に分離）
                bet_mask = rr_valid & (y_pred * np.nan_to_num(rr_odds) >= rr_min_ev)
                if not bet_mask.any():
                    return 0.0
                total_bet = float(bet_mask.sum()) * 100.0
                total_return = 100.0 * rr_payout[bet_mask & rr_hit].sum()
                return (total_return / total_bet * 100.0) if total_bet > 0 else 0.0

            return ModelEvaluator.compute_auc(y_val, y_pred)

        from optuna.samplers import TPESampler

        sampler = TPESampler(n_startup_trials=20, multivariate=True, seed=42)
        # trial.report() を呼ばない設計のため MedianPruner は機能しない。
        # 誤解を避けるためプルーナーは設定しない。
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"[{model_name}] 最適なパラメータ: {study.best_params}")
        logger.info(f"[{model_name}] 最良スコア: {study.best_value:.4f}")

        return study.best_params

    def hyperparameter_tuning_ranking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: List[int],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        groups_val: List[int],
        n_trials: int = 100,
    ) -> Dict:
        """Optunaを使用したランキングモデルのハイパーパラメータチューニング。

        LambdaRank（NDCG@1最大化）で num_leaves / learning_rate 等を探索する。
        eval_at / label_gain は config の lightgbm_ranking セクションから取得する。

        Args:
            X_train: 学習データ特徴量
            y_train: 学習ラベル（非負整数スコア）
            groups_train: 学習セットのレースごとの馬数リスト
            X_val: 検証データ特徴量
            y_val: 検証ラベル
            groups_val: 検証セットのレースごとの馬数リスト
            n_trials: Optuna 試行回数

        Returns:
            Dict: 最適なパラメータ（空辞書の場合は最適化失敗）
        """
        logger.info("ランキングモデル ハイパーパラメータチューニング開始")

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.error("Optunaがインストールされていません")
            return {}

        ranking_cfg = self.config.get('lightgbm_ranking', {})
        label_gain = ranking_cfg.get('label_gain', [0, 1, 3, 7, 15])
        # objective は ndcg@1 / ndcg@3 を参照するため、eval_at に 1,3 を必ず含める。
        # 含まれないと best_score から取得できず黙って 0.0 になり最適化が無意味化する。
        eval_at = sorted(set(ranking_cfg.get('eval_at', [1])) | {1, 3})

        # Dataset は試行ごとに再構築せず1回だけ構築して再利用する。
        # min_data_in_leaf 等をtrialで変えるため feature_pre_filter は無効化必須。
        train_data = lgb.Dataset(
            X_train, label=y_train, group=groups_train, free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, group=groups_val, reference=train_data, free_raw_data=False
        )

        def objective(trial: 'optuna.Trial') -> float:  # type: ignore[name-defined]
            boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
            params = {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'eval_at': eval_at,
                'label_gain': label_gain,
                'feature_pre_filter': False,
                'boosting_type': boosting_type,
                'verbose': -1,
                'seed': 42,
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 150),
                'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 15.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 15.0, log=True),
                'max_depth': trial.suggest_int('max_depth', 4, 12),
            }
            if boosting_type == 'dart':
                params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.3)
                params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

            if boosting_type == 'dart':
                num_boost_round = trial.suggest_int('n_estimators', 200, 1000)
                callbacks_list = [lgb.log_evaluation(period=0)]
            else:
                num_boost_round = 1000
                callbacks_list = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ]

            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                valid_names=['valid'],
                callbacks=callbacks_list,
            )
            ndcg1 = model.best_score['valid'].get('ndcg@1', 0.0)
            ndcg3 = model.best_score['valid'].get('ndcg@3', 0.0)
            return 0.7 * ndcg1 + 0.3 * ndcg3

        from optuna.samplers import TPESampler

        sampler = TPESampler(n_startup_trials=20, multivariate=True, seed=42)
        # trial.report() を呼ばない設計のため MedianPruner は機能しない。設定しない。
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info(f"最適なパラメータ: {study.best_params}")
        logger.info(f"最良スコア (0.7*NDCG@1 + 0.3*NDCG@3): {study.best_value:.4f}")

        return study.best_params

    # ------------------------------------------------------------------
    # OOF スタッキング
    # ------------------------------------------------------------------

    def build_oof_meta_features(self, fold_data: List[Dict]) -> pd.DataFrame:
        """
        blend 最適化で収集した fold_data から OOF メタ特徴量 DataFrame を生成する。

        Args:
            fold_data: optimize_blend_alpha_cv() が収集したフォールドデータのリスト。
                各要素は win_proba / ranking_scores / race_id / odds_win / finish_position を含む。

        Returns:
            pd.DataFrame: win_proba / ranking_score / odds_win / finish_position / target_win を持つ DataFrame
        """
        frames = []
        for fd in fold_data:
            # target_win は apply を廃止しベクトルで生成
            won1 = fd.get('won1')
            if won1 is None:
                fin = np.asarray(fd['finish_position'])
                won1 = (~pd.isna(fin)) & (np.nan_to_num(fin, nan=-1).astype(int) == 1)
            df_tmp = pd.DataFrame({
                'win_proba': fd['win_proba'],
                'ranking_score': fd['ranking_scores'],
                'odds_win': fd['odds_win'],
                'finish_position': fd['finish_position'],
                'target_win': np.asarray(won1).astype(int),
            })
            frames.append(df_tmp)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def train_meta_model(
        self,
        oof_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> object:
        """
        OOF 予測を入力とするメタモデル（LogisticRegression）を学習する。

        Args:
            oof_df: build_oof_meta_features() が返す DataFrame
            feature_cols: メタ特徴量として使う列名（デフォルト: win_proba / ranking_score）

        Returns:
            sklearn の LogisticRegression インスタンス（学習済み）
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        if oof_df.empty:
            logger.warning("OOF DataFrame が空のためメタモデル学習をスキップします")
            return None

        cols = feature_cols or ['win_proba', 'ranking_score']
        cols = [c for c in cols if c in oof_df.columns]
        if not cols:
            logger.warning("メタ特徴量カラムが見つかりません。スキップします")
            return None

        X_meta = oof_df[cols].values
        y_meta = oof_df['target_win'].values

        meta_model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ])
        meta_model.fit(X_meta, y_meta)

        pos_rate = y_meta.mean()
        logger.info(
            f"メタモデル学習完了 (n={len(y_meta)}, pos_rate={pos_rate:.3f}, "
            f"features={cols})"
        )
        self._meta_model: object = meta_model
        self._meta_feature_cols: List[str] = cols
        return meta_model

    def predict_meta_model(self, win_proba: np.ndarray, ranking_scores: np.ndarray) -> np.ndarray:
        """
        学習済みメタモデルで確率を予測する。

        Args:
            win_proba: binary モデルの予測確率
            ranking_scores: ranking モデルのスコア

        Returns:
            np.ndarray: メタモデルの予測確率（正例確率）

        Raises:
            RuntimeError: メタモデルが未学習の場合。
                以前は win_proba を黙って返すフォールバックだったが、
                呼び出し側が「メタモデルの出力」を期待しているのに別物（binary 生確率）が
                返ると誤った確率で判断が進むため、想定外として明示的に停止する
                （CLAUDE.md「想定外は止めてエラー」方針）。
        """
        meta_model = getattr(self, '_meta_model', None)
        meta_cols = getattr(self, '_meta_feature_cols', ['win_proba', 'ranking_score'])
        if meta_model is None:
            raise RuntimeError(
                "メタモデルが未学習です。predict_meta_model の前に train_meta_model を実行してください。"
            )

        feature_map = {'win_proba': win_proba, 'ranking_score': ranking_scores}
        X_meta = np.column_stack([feature_map[c] for c in meta_cols])
        return meta_model.predict_proba(X_meta)[:, 1]
