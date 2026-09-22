"""
モデル生成モジュール（単独実行エントリポイント）

    python -m src.models.model_creator

    # バイナリ（win/place）のみ
    python -m src.models.model_creator --model-type binary

    # ランキングのみ
    python -m src.models.model_creator --model-type ranking

    # 馬連のみ（候補絞り込みに必要な単勝モデルも併せて学習する）
    python -m src.models.model_creator --model-type umaren

デフォルトでは --model-type all 相当の動作となり、
win / place / ranking を学習・保存する（enabled_models に 'umaren' を
含めると馬連モデルも学習する）。

ここで定義する ModelCreator は、単独実行ルーチン（run_train）が
LightGBMTrainer / LightGBMOptimizer / ModelEvaluator を扱うための
最小限のファサードであり、run_train が必要とする操作のみを公開する。
"""
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

from src.cli_common import derive_model_feature_cols, save_model_feature_schema
from src.models.evaluation import ModelEvaluator
from src.models.odds_series import (
    BET_ODDS_COLS,
    attach_meta_columns,
    attach_umaren_meta_columns,
    build_umaren_payout,
    has_two_source_odds,
    load_payouts,
)
from src.models.optimizer import LightGBMOptimizer
from src.models.trainer import LightGBMTrainer

logger = logging.getLogger(__name__)


class ModelCreator:
    """
    単独実行ルーチン（run_train）が models/ パッケージを操作するための
    最小限のファサードクラス。

    LightGBMTrainer / LightGBMOptimizer / ModelEvaluator を内部で管理し、
    run_train が必要とする学習・最適化・評価の操作のみを公開する。
    """

    def __init__(self, config: dict, features_def: dict | None = None):
        """
        初期化

        Args:
            config: 全体設定辞書（config.yaml のルート辞書）
            features_def: 特徴量定義辞書（features.json の内容）。省略時は空辞書。
        """
        self._trainer = LightGBMTrainer(config.get('model', {}), features_def)
        self._optimizer = LightGBMOptimizer(self._trainer, config)
        self._evaluator = ModelEvaluator(config.get('evaluation', {}))

    # ------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------

    @property
    def models(self) -> Dict[str, lgb.Booster]:
        """学習済みモデルの辞書。"""
        return self._trainer.models

    @property
    def profit_iterations(self) -> Dict[str, int]:
        """モデル名 → 回収率最良イテレーションの辞書。"""
        return self._trainer.profit_iterations

    @property
    def min_ranking_rank(self) -> int:
        """賭け対象とするランキング順位の上限（0 は無制限）。"""
        return self._evaluator.min_ranking_rank

    # ------------------------------------------------------------------
    # 学習系（LightGBMTrainer に委譲）
    # ------------------------------------------------------------------

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        test_size_months: int = 6,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        データを学習用・テスト用に分割する。

        Args:
            df: データフレーム
            feature_cols: 特徴量カラムのリスト
            target_col: ターゲットカラム名
            test_size_months: テストデータの期間（月）

        Returns:
            Tuple: (X_train, X_test, y_train, y_test, train_dates)
        """
        return self._trainer.prepare_data(df, feature_cols, target_col, test_size_months)

    @staticmethod
    def compute_time_decay_weights(
        dates: pd.Series,
        recent_years: float = 2.0,
        mid_years: float = 4.0,
        w_recent: float = 3.0,
        w_mid: float = 2.0,
    ) -> np.ndarray:
        """
        時間減衰サンプルウェイトを計算する。

        Args:
            dates: 各サンプルのレース日付
            recent_years: 直近データとみなす年数
            mid_years: 中間データとみなす年数
            w_recent: 直近データのウェイト
            w_mid: 中間データのウェイト

        Returns:
            np.ndarray: 各サンプルのサンプルウェイト
        """
        return LightGBMTrainer.compute_time_decay_weights(
            dates,
            recent_years=recent_years,
            mid_years=mid_years,
            w_recent=w_recent,
            w_mid=w_mid,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_name: str = 'model',
        sample_weight: Optional[np.ndarray] = None,
        val_df_for_profit: Optional[pd.DataFrame] = None,
        fobj_beta: Optional[float] = None,
    ) -> lgb.Booster:
        """
        バイナリ分類モデルを学習する。

        Args:
            X_train: 学習データの特徴量
            y_train: 学習データのターゲット
            X_val: 検証データの特徴量（オプション）
            y_val: 検証データのターゲット（オプション）
            model_name: モデル名
            sample_weight: サンプルウェイト（オプション）
            val_df_for_profit: 回収率コールバック用の検証メタ DataFrame（オプション）
            fobj_beta: Phase 2b 用。custom fobj の非対称化強度（None で無効）

        Returns:
            lgb.Booster: 学習済みモデル
        """
        return self._trainer.train(
            X_train, y_train, X_val, y_val, model_name, sample_weight,
            val_df_for_profit=val_df_for_profit, fobj_beta=fobj_beta,
        )

    @staticmethod
    def compute_odds_weights(
        odds: np.ndarray,
        gamma: float = 0.5,
        w_min: float = 0.5,
        w_max: float = 5.0,
    ) -> np.ndarray:
        """オッズ由来のサンプルウェイトを計算する（高オッズ的中を重視）。

        Args:
            odds: 締切前オッズの配列（確定払戻を渡してはならない）
            gamma: 重み付けの強さ（0 で等重み）
            w_min: ウェイトの下限
            w_max: ウェイトの上限

        Returns:
            np.ndarray: オッズウェイト
        """
        return LightGBMTrainer.compute_odds_weights(odds, gamma, w_min, w_max)

    def calibrate_model(
        self,
        model_name: str,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = 'isotonic',
        min_samples_per_bin: int = 0,
        min_pos_per_bin: int = 0,
    ) -> None:
        """
        学習済みモデルの出力確率をキャリブレーションする。

        Args:
            model_name: キャリブレーション対象のモデル名
            X_cal: キャリブレーション用特徴量
            y_cal: キャリブレーション用ラベル
            method: 'isotonic'（デフォルト）または 'sigmoid'
            min_samples_per_bin: isotonic の1プラトーあたり最小標本数（0 で無効）
            min_pos_per_bin: isotonic の1ビンあたり最小正例数（0 で無効）
        """
        self._trainer.calibrate_model(
            model_name, X_cal, y_cal, method, min_samples_per_bin,
            min_pos_per_bin=min_pos_per_bin,
        )

    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = 'model',
        use_profit_iter: bool = False,
    ) -> np.ndarray:
        """
        予測を実行する。

        Args:
            X: 特徴量データ
            model_name: モデル名
            use_profit_iter: True のとき ranking モデルで回収率最良イテレーションを使う
                （記録が無ければ best_iteration にフォールバック）。

        Returns:
            np.ndarray: 予測確率（またはランキングスコア）
        """
        return self._trainer.predict(X, model_name, use_profit_iter=use_profit_iter)

    def save_model(self, model_name: str = 'model', filename: Optional[str] = None) -> None:
        """
        モデルをファイルに保存する。

        Args:
            model_name: モデル名
            filename: 保存ファイル名
        """
        self._trainer.save_model(model_name, filename)

    def load_model(self, filename: str, model_name: str = 'model') -> None:
        """
        保存済みモデルを読み込む（キャリブレータ・メタも対で復元される）。

        Args:
            filename: model_dir 配下のファイル名
            model_name: 読み込み先のモデル名
        """
        self._trainer.load_model(filename, model_name)

    @property
    def model_dir(self) -> Path:
        """モデルの保存・読込ディレクトリ。"""
        return self._trainer.model_dir

    def update_lgbm_params(self, section_key: str, params: Dict) -> None:
        """次の学習で使う LightGBM パラメータセクションを更新する。

        Optuna で得た最適パラメータを、trainer が参照する設定辞書
        （config['model'] 相当）の該当セクションに反映する。
        呼び出し側が private 属性 `_trainer` に直接触れずに済むよう公開する。

        Args:
            section_key: 'lightgbm' / 'lightgbm_place' / 'lightgbm_ranking'
            params: 更新するパラメータ辞書
        """
        self._trainer.config.setdefault(section_key, {}).update(params)

    def get_feature_importance(
        self,
        model_name: str = 'model',
        importance_type: str = 'gain',
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        特徴量重要度を取得する。

        Args:
            model_name: モデル名
            importance_type: 重要度のタイプ（'gain', 'split'）
            top_n: 上位N件を取得

        Returns:
            pd.DataFrame: 特徴量重要度
        """
        return self._trainer.get_feature_importance(model_name, importance_type, top_n)

    def prepare_ranking_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ranking_label_config: dict,
        test_size_months: int = 6,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
        List[int], List[int], pd.DataFrame, pd.DataFrame, List[str]
    ]:
        """
        ランキング学習用データを準備する。

        Args:
            df: 特徴量DataFrame（race_id, date, finish_position カラムが必須）
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label']
            test_size_months: テスト期間（月）

        Returns:
            (X_train, X_test, y_train, y_test,
             groups_train, groups_test, train_df, test_df, ranking_feature_cols)
        """
        return self._trainer.prepare_ranking_data(
            df, feature_cols, ranking_label_config, test_size_months
        )

    def train_ranking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: List[int],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        groups_val: Optional[List[int]] = None,
        model_name: str = 'ranking',
        val_df_for_profit: Optional[pd.DataFrame] = None,
        val_feature_cols_for_profit: Optional[List[str]] = None,
    ) -> lgb.Booster:
        """
        ランキングモデル（LambdaRank）を学習する。

        Args:
            X_train: 学習特徴量
            y_train: 関連度スコア（非負整数）
            groups_train: 各レースの馬数リスト
            X_val: 検証特徴量（オプション）
            y_val: 検証ラベル（オプション）
            groups_val: 検証グループリスト（オプション）
            model_name: モデル識別子
            val_df_for_profit: 利益コールバック用の検証DataFrame（オプション）
            val_feature_cols_for_profit: 利益コールバック用の特徴量カラム（オプション）

        Returns:
            lgb.Booster: 学習済みランキングモデル
        """
        return self._trainer.train_ranking(
            X_train, y_train, groups_train,
            X_val, y_val, groups_val,
            model_name, val_df_for_profit, val_feature_cols_for_profit,
        )

    # ------------------------------------------------------------------
    # 最適化・CV系（LightGBMOptimizer に委譲）
    # evaluator は自動注入するため呼び出し側での受け渡しが不要
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
        bet_type: str = 'win',
        threshold: float = 1.05,
    ) -> Dict:
        """
        日付ベースの時系列交差検証を実行する。

        evaluator は内部で自動注入されるため、呼び出し側での受け渡しは不要。

        Args:
            df: データフレーム（'date' カラム必須）
            feature_cols: 特徴量カラム
            target_col: ターゲットカラム（'target_win' など）
            n_splits: フォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            train_window_months: 固定窓の学習幅（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding'（拡張窓）or 'rolling'（固定窓）
            bet_type: 賭けの種類（'win' / 'place'）
            threshold: 回収率計算に使う最低期待値

        Returns:
            Dict: fold_results リストと集計統計
        """
        return self._optimizer.cross_validate(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            n_splits=n_splits,
            min_train_months=min_train_months,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            cv_type=cv_type,
            evaluator=self._evaluator,
            bet_type=bet_type,
            threshold=threshold,
        )

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
    ) -> Dict:
        """
        ランキングモデルの時系列交差検証を実行する。

        evaluator は内部で自動注入されるため、呼び出し側での受け渡しは不要。

        Args:
            df: データフレーム（'date', 'race_id', 'finish_position' カラム必須）
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label']
            n_splits: フォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            train_window_months: 固定窓の学習幅（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding' or 'rolling'

        Returns:
            Dict: fold_results リストと集計統計
        """
        return self._optimizer.cross_validate_ranking(
            df=df,
            feature_cols=feature_cols,
            ranking_label_config=ranking_label_config,
            n_splits=n_splits,
            min_train_months=min_train_months,
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            cv_type=cv_type,
            evaluator=self._evaluator,
        )

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
        CVベースで blend_alpha / ranking_temperature を Optuna で最適化する。

        Args:
            df: 特徴量DataFrame
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label']
            min_ev_win: 単勝購入の最低期待値
            min_ev_place: 複勝購入の最低期待値
            n_splits: CVフォールド数
            min_train_months: 拡張窓の最小学習期間（月）
            test_window_months: テストウィンドウ幅（月）
            cv_type: 'expanding' or 'rolling'
            n_trials: Optuna 試行回数

        Returns:
            Dict: best_alpha, best_alpha_place, best_temperature, best_recovery_rate
        """
        return self._optimizer.optimize_blend_alpha_cv(
            df=df,
            feature_cols=feature_cols,
            ranking_label_config=ranking_label_config,
            min_ev_win=min_ev_win,
            min_ev_place=min_ev_place,
            n_splits=n_splits,
            min_train_months=min_train_months,
            test_window_months=test_window_months,
            cv_type=cv_type,
            n_trials=n_trials,
        )

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
        Optunaを使用してバイナリモデルのハイパーパラメータを最適化する。

        Args:
            X_train: 学習データ
            y_train: 学習ラベル
            X_val: 検証データ
            y_val: 検証ラベル
            n_trials: 試行回数
            model_name: モデル識別子（'win' or 'place'）。
                'place' のときは lightgbm_place セクションをベースに最適化する。
            val_meta_df: 回収率計算用の検証メタデータ。
                締切前オッズ（odds_pre_*）・確定払戻（payout_*）・的中判定列が必須。
            objective_metric: 最適化指標（'auc' または 'recovery_rate'）。
            fobj_sample_weight: Phase 2b 用。指定すると各 trial で custom fobj の
                beta を探索する（``fobj_beta_range`` も必要）。
            fobj_beta_range: Phase 2b 用。beta の探索範囲 ``(low, high)``。

        Returns:
            Dict: 最適なパラメータ（fobj 探索時は ``fobj_beta`` を含む）
        """
        return self._optimizer.hyperparameter_tuning(
            X_train, y_train, X_val, y_val, n_trials, model_name,
            val_meta_df=val_meta_df, objective_metric=objective_metric,
            fobj_sample_weight=fobj_sample_weight, fobj_beta_range=fobj_beta_range,
        )

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
        """
        Optunaを使用してランキングモデルのハイパーパラメータを最適化する。

        Args:
            X_train: 学習データ特徴量
            y_train: 学習ラベル（非負整数スコア）
            groups_train: 学習セットのレースごとの馬数リスト
            X_val: 検証データ特徴量
            y_val: 検証ラベル
            groups_val: 検証セットのレースごとの馬数リスト
            n_trials: Optuna 試行回数

        Returns:
            Dict: 最適なパラメータ
        """
        return self._optimizer.hyperparameter_tuning_ranking(
            X_train, y_train, groups_train, X_val, y_val, groups_val, n_trials
        )

    # ------------------------------------------------------------------
    # 評価系（ModelEvaluator に委譲）
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        y_pred_binary: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        モデルの性能を評価する。

        Args:
            y_true: 正解ラベル
            y_pred_proba: 予測確率
            threshold: 閾値（Accuracy / Precision / Recall / F1 を計算するためだけに使用）
            y_pred_binary: 外部で計算済みの二値予測（指定時は threshold より優先）

        Returns:
            Dict: 評価指標の辞書
        """
        return self._evaluator.evaluate_model(y_true, y_pred_proba, threshold, y_pred_binary)

    def calculate_recovery_rate(
        self,
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        min_ev: float,
        bet_type: str = 'win',
    ) -> Dict:
        """
        回収率を計算する（期待値ベース）。

        Args:
            df: データフレーム（オッズ情報を含む）
            y_pred_proba: 予測確率
            min_ev: 購入する最低期待値
            bet_type: 賭けの種類（'win' or 'place'）

        Returns:
            Dict: 回収率と統計情報
        """
        return self._evaluator.calculate_recovery_rate(df, y_pred_proba, min_ev, bet_type)

    def print_summary(self, metrics: Dict, recovery_results: Dict) -> None:
        """
        評価結果のサマリーを表示する。

        Args:
            metrics: モデル評価指標
            recovery_results: 回収率結果
        """
        self._evaluator.print_summary(metrics, recovery_results)


# ======================================================================
# 単独実行エントリポイント（学習ルーチン）
#
# self に依存しないモジュール関数として学習処理一式を提供する。これにより
#   python -m src.models.model_creator
# だけでモデル生成が完結する。
# ======================================================================

_LABEL_MAP = {'win': '単勝', 'place': '複勝'}

# Optuna で得た LightGBM パラメータのうち config に書き戻す対象キー。
# objective / metric / is_unbalance 等の固定パラメータは変更しない。
_LGBM_TUNABLE_KEYS: frozenset = frozenset({
    'num_leaves', 'learning_rate', 'feature_fraction',
    'bagging_fraction', 'bagging_freq', 'min_data_in_leaf',
    'lambda_l1', 'lambda_l2',
})


def _apply_updates_to_node(root: dict, updates: Dict) -> None:
    """ドット区切りキーの updates を root（ネスト辞書）にインプレース適用する。"""
    for dotted_key, value in updates.items():
        keys = dotted_key.split('.')
        node = root
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value


def _update_config_yaml(config: dict, config_path: str, updates: Dict) -> None:
    """config.yaml の指定キーをメモリ（config）と同時に更新する。

    updates の形式: {'section.key': value}
    例: {'evaluation.blend_alpha': 0.6}

    config.yaml の書き戻しは ruamel.yaml の round-trip モードで行い、
    既存のコメント・キー順・引用形式を保持する。ruamel.yaml が利用できない環境では
    従来の yaml.dump にフォールバックする（この場合コメントは失われるため警告する）。

    Args:
        config: メモリ上の設定辞書（インプレース更新される）
        config_path: config.yaml のパス
        updates: ドット区切りのキーパスと更新値の辞書
    """
    # メモリ上の設定は常に更新する
    _apply_updates_to_node(config, updates)

    path = Path(config_path)
    if not path.exists():
        logger.warning(f"config.yaml が見つかりません: {path}（メモリ上のみ更新）")
        return

    try:
        from ruamel.yaml import YAML

        yaml_rt = YAML()  # round-trip モード（コメント・順序を保持）
        yaml_rt.preserve_quotes = True
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml_rt.load(f) or {}
        _apply_updates_to_node(raw_config, updates)
        with open(path, 'w', encoding='utf-8') as f:
            yaml_rt.dump(raw_config, f)
        logger.info("config.yaml を更新しました（コメント保持）: %s", path)
    except ImportError:
        # ruamel.yaml 未導入時のフォールバック（コメントは失われる）
        logger.warning(
            "ruamel.yaml が見つからないため yaml.dump で書き戻します。"
            "config.yaml のコメントが失われます（requirements.txt の ruamel.yaml を導入してください）。"
        )
        with open(path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f) or {}
        _apply_updates_to_node(raw_config, updates)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(raw_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        logger.info("config.yaml を更新しました: %s", path)


def _write_lgbm_params_to_config(
    config: dict, config_path: str, best_params: Dict, model_name: str
) -> None:
    """Optuna で得た LightGBM 最適パラメータをメモリと config.yaml に書き戻す。

    チューニング対象キー（_LGBM_TUNABLE_KEYS）のみを更新する。

    Args:
        config: メモリ上の設定辞書
        config_path: config.yaml のパス
        best_params: hyperparameter_tuning() / hyperparameter_tuning_ranking() の戻り値
        model_name: 'win'、'place'、'ranking'、または 'umaren'
    """
    filtered = {k: v for k, v in best_params.items() if k in _LGBM_TUNABLE_KEYS}
    if not filtered:
        return
    section_map = {
        'win': 'lightgbm',
        'place': 'lightgbm_place',
        'ranking': 'lightgbm_ranking',
        'umaren': 'lightgbm_umaren',
    }
    section_key = section_map.get(model_name, 'lightgbm')
    logger.info(f"[{model_name}] 最適 LightGBM パラメータ: {filtered}")
    _update_config_yaml(
        config, config_path, {f"model.{section_key}.{k}": v for k, v in filtered.items()}
    )


def _resolve_calibration_config(config: dict, model_name: str) -> dict:
    """券種ごとのキャリブレーション設定を解決する。

    ``model.calibration`` のグローバル値を、``model.calibration.<model_name>``
    セクションに書いたキーだけが上書きする（``_resolve_fobj_beta`` と同じ規則）。

    券種別に分ける必要があるのは、正例率が券種で桁違いに異なるためである。
    馬連は top_k=6 で 1レース15ペア中1的中（正例率 ~6.7%）と低く、
    isotonic + 等頻度ビン集約では下位ビンの正例が0件になって
    「出力が厳密に 0.0」の広いプラトーが生まれる。EV = proba × odds は
    乗法的なので、0.0 になったペアはオッズが何倍でも永久に選ばれず、
    モデルの順序情報がそこで完全に失われる。
    詳細: output/20260811_06_umaren_zero_proba_investigation.md

    Args:
        config: 設定辞書
        model_name: 'win' / 'place' / 'umaren'

    Returns:
        dict: マージ済みのキャリブレーション設定
    """
    cal_cfg = config.get('model', {}).get('calibration', {})
    type_cfg = cal_cfg.get(model_name, {})
    if not isinstance(type_cfg, dict):
        raise ValueError(
            f"model.calibration.{model_name} は辞書である必要があります: {type_cfg!r}"
        )
    return {**cal_cfg, **type_cfg}


def _fit_calibrator(
    config: dict,
    creator: 'ModelCreator',
    model_name: str,
    df: pd.DataFrame,
    split_date: pd.Timestamp,
    feature_cols: List[str],
    target_col: str,
) -> None:
    """学習データ末尾のホールドアウトで calibrator を fit する。

    テストセット（``date >= split_date``）で fit するとキャリブレータが評価データの
    実測正例率に張り付き、回収率が楽観バイアスになるため使わない。

    IMPORTANT:
        期間が短いと isotonic が過学習し、出力確率が少数の離散値に量子化されて
        EV が系統的に過大評価される。既定を 12ヶ月としたのはこのため
        （`model.calibration.months` で変更可能）。

    設定は `model.calibration` のグローバル値を `model.calibration.<model_name>`
    セクションが上書きする（`_resolve_calibration_config`）。正例率が低い馬連は
    `method: sigmoid` を選ぶことで isotonic の 0.0 プラトー問題を回避する。

    Args:
        config: 設定辞書
        creator: 学習済みモデルを保持する ModelCreator
        model_name: キャリブレーション対象のモデル名
        df: 全期間のデータ（'date' 列必須）
        split_date: この日付以降はテストデータ
        feature_cols: 特徴量カラム
        target_col: 目的変数カラム
    """
    cal_cfg = _resolve_calibration_config(config, model_name)
    months = int(cal_cfg.get('months', 12))
    method = cal_cfg.get('method', 'isotonic')
    min_samples_per_bin = int(cal_cfg.get('min_samples_per_bin', 0))
    min_samples = int(cal_cfg.get('min_samples', 0))
    min_pos_per_bin = int(cal_cfg.get('min_pos_per_bin', 0))

    cal_cutoff = split_date - pd.DateOffset(months=months)
    cal_df = df[(df['date'] >= cal_cutoff) & (df['date'] < split_date)]

    if len(cal_df) == 0 or cal_df[target_col].sum() == 0:
        logger.warning(
            f"[{model_name}] キャリブレーション用ホールドアウトが不足のためスキップします"
        )
        return

    logger.info(
        f"[{model_name}] キャリブレーション期間: {months}ヶ月 "
        f"({cal_cutoff.date()} 〜 {split_date.date()}) {len(cal_df)} 件 method={method}"
    )
    if min_samples > 0 and len(cal_df) < min_samples:
        logger.warning(
            f"[{model_name}] キャリブレーション標本が {len(cal_df)} 件しかありません"
            f"（推奨 {min_samples} 件以上）。確率が量子化され EV が歪む恐れがあります。"
        )

    creator.calibrate_model(
        model_name,
        cal_df[list(feature_cols)],
        cal_df[target_col],
        method=method,
        min_samples_per_bin=min_samples_per_bin,
        min_pos_per_bin=min_pos_per_bin,
    )


def _load_features_for_training(config: dict, feature_file: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    """特徴量ファイルをロードし、モデル特徴量設定を保存して (df, model_feature_cols) を返す。

    回収率評価用のメタ列（締切前オッズ・確定払戻）も結合する。
    メタ列は model_feature_cols を確定させた**後**に結合するため、
    特徴量として混入することはない。

    Args:
        config: 設定辞書
        feature_file: 特徴量ファイル名（省略時は 'features.feather'）
    """
    from src.features.feature_engineering import FeatureEngineer

    features_def = config.get('features_def') or {}
    engineer = FeatureEngineer(config['data'], features_def)
    df = engineer.load_processed_data(feature_file or 'features.feather', file_format='feather')
    # model_keep_cols（race_id / date / target_* / finish_position / odds_*）以外を特徴量とみなす
    model_feature_cols = derive_model_feature_cols(df.columns.tolist(), features_def)
    schema_path = save_model_feature_schema(config, model_feature_cols)
    logger.info(f"モデル特徴量スキーマを保存: {schema_path} ({len(model_feature_cols)} 特徴量)")

    # 評価専用メタ列は特徴量確定後に結合する（model_feature_cols には入らない）
    df = attach_meta_columns(df, config, bet_types=('win', 'place'))
    return df, model_feature_cols


def _run_cv_evaluation(
    config: dict, creator: 'ModelCreator', df: pd.DataFrame, feature_cols: List[str], cv_config: dict
) -> Dict:
    """学習前に時系列CVを実行し、フォールドごとの結果をログ出力する（binary）。"""
    from src.models.cv_reporter import log_cv_fold_table

    n_splits = cv_config.get('n_splits', 5)
    min_train_months = cv_config.get('min_train_months', 12)
    train_window_months = cv_config.get('train_window_months', 12)
    test_window_months = cv_config.get('test_window_months', 3)
    cv_type = cv_config.get('cv_type', 'expanding')

    eval_cfg = config.get('evaluation', {})
    targets = [
        ('target_win', 'win', eval_cfg.get('min_ev_win', 1.05)),
        ('target_place', 'place', eval_cfg.get('min_ev_place', 1.05)),
    ]
    cv_types_to_run = ['expanding', 'rolling'] if cv_type == 'both' else [cv_type]

    collected: Dict = {}
    for run_type in cv_types_to_run:
        type_label = '拡張窓（Expanding）' if run_type == 'expanding' else '固定窓（Rolling）'
        logger.info("=" * 60)
        logger.info(f"=== 時系列クロスバリデーション: {type_label} ===")
        logger.info("=" * 60)
        collected[run_type] = {}
        for target_col, bet_type, threshold in targets:
            logger.info(f"\n--- Target: {target_col} ---")
            cv_result = creator.cross_validate(
                df=df, feature_cols=feature_cols, target_col=target_col,
                n_splits=n_splits, min_train_months=min_train_months,
                train_window_months=train_window_months, test_window_months=test_window_months,
                cv_type=run_type, bet_type=bet_type, threshold=threshold,
            )
            fold_results = cv_result.get('fold_results', [])
            if not fold_results:
                logger.info("有効なフォールドがありませんでした")
                continue
            log_cv_fold_table(fold_results, cv_result)
            collected[run_type][bet_type] = cv_result

    logger.info("=" * 60)
    logger.info("=== CV完了。最終モデル学習を開始します ===")
    logger.info("=" * 60)
    return collected


def _run_cv_evaluation_ranking(
    config: dict, creator: 'ModelCreator', df: pd.DataFrame, feature_cols: List[str], cv_config: dict
) -> Dict:
    """ランキングモデルの時系列CVを実行し、フォールドごとの結果をログ出力する。"""
    from src.models.cv_reporter import log_ranking_cv_fold_table

    ranking_label_config = config.get('model', {}).get('ranking_label', {
        'top1': 2, 'top3': 1, 'other': 0
    })
    n_splits = cv_config.get('n_splits', 5)
    min_train_months = cv_config.get('min_train_months', 12)
    train_window_months = cv_config.get('train_window_months', 12)
    test_window_months = cv_config.get('test_window_months', 3)
    cv_type = cv_config.get('cv_type', 'expanding')
    cv_types_to_run = ['expanding', 'rolling'] if cv_type == 'both' else [cv_type]

    collected: Dict = {}
    for run_type in cv_types_to_run:
        type_label = '拡張窓（Expanding）' if run_type == 'expanding' else '固定窓（Rolling）'
        logger.info("=" * 60)
        logger.info(f"=== ランキング時系列クロスバリデーション: {type_label} ===")
        logger.info("=" * 60)
        cv_result = creator.cross_validate_ranking(
            df=df, feature_cols=feature_cols, ranking_label_config=ranking_label_config,
            n_splits=n_splits, min_train_months=min_train_months,
            train_window_months=train_window_months, test_window_months=test_window_months,
            cv_type=run_type,
        )
        fold_results = cv_result.get('fold_results', [])
        if not fold_results:
            logger.info("有効なフォールドがありませんでした")
            continue
        log_ranking_cv_fold_table(fold_results, cv_result)
        collected[run_type] = cv_result
    return collected


def _resolve_fobj_beta(config: dict, bet_type: str) -> Optional[float]:
    """Phase 2b の custom fobj を使うかどうかと非対称化強度 beta を解決する。

    ``model.odds_weight.enabled`` が True であることが前提（オッズ重み機構自体の
    グローバル ON/OFF）。その上で ``fobj_enabled`` / ``fobj_beta`` は
    ``model.odds_weight.<bet_type>`` セクションでの個別上書きを許す
    （win は fobj、place は sample_weight、のような使い分けができる）。
    ``{**ow_cfg, **ow_cfg[bet_type]}`` でマージした**後**の値を見て判定することで、
    bet_type セクションでの ``fobj_enabled`` 上書きが正しく効くようにする。

    Args:
        config: 設定辞書
        bet_type: 'win' / 'place' / 'umaren'

    Returns:
        Optional[float]: custom fobj を使う場合は beta（0 以上）、使わない場合 None
    """
    ow_cfg = config.get('model', {}).get('odds_weight', {})
    if not ow_cfg.get('enabled', False):
        return None
    type_cfg = {**ow_cfg, **ow_cfg.get(bet_type, {})}
    if not type_cfg.get('fobj_enabled', False):
        return None
    return float(type_cfg.get('fobj_beta', 0.0))


def _resolve_fobj_beta_search_range(config: dict, bet_type: str) -> Optional[Tuple[float, float]]:
    """Optuna HPO 内で custom fobj の beta を探索するかどうかを解決する。

    ``_resolve_fobj_beta`` と同じマージ規則（グローバル値を bet_type セクションで
    上書き可能）で ``fobj_beta_search`` / ``fobj_beta_range`` を解決する。
    fobj 自体が無効（``_resolve_fobj_beta`` が None を返す設定）なら、
    ``fobj_beta_search: true`` であっても探索しない
    （HPO は最終学習で使われる beta の探索なので、fobj が無効なら意味がない）。

    Args:
        config: 設定辞書
        bet_type: 'win' / 'place' / 'umaren'

    Returns:
        Optional[Tuple[float, float]]: 探索する場合は ``(low, high)``、しない場合 None

    Raises:
        ValueError: ``fobj_beta_range`` の要素数が2でない、または low > high の場合
    """
    if _resolve_fobj_beta(config, bet_type) is None:
        return None
    ow_cfg = config.get('model', {}).get('odds_weight', {})
    type_cfg = {**ow_cfg, **ow_cfg.get(bet_type, {})}
    if not type_cfg.get('fobj_beta_search', False):
        return None

    beta_range = type_cfg.get('fobj_beta_range', [0.0, 3.0])
    if len(beta_range) != 2:
        raise ValueError(
            f"[{bet_type}] fobj_beta_range は [low, high] の2要素である必要があります: {beta_range}"
        )
    low, high = float(beta_range[0]), float(beta_range[1])
    if low > high:
        raise ValueError(f"[{bet_type}] fobj_beta_range は low <= high である必要があります: {beta_range}")
    return (low, high)


def _apply_odds_weights(
    config: dict,
    creator: 'ModelCreator',
    sample_weight: np.ndarray,
    train_meta: pd.DataFrame,
    bet_type: str,
) -> np.ndarray:
    """時間減衰ウェイトにオッズ重み（高オッズ的中を重視）を掛け合わせる。

    ``model.odds_weight.enabled`` が False（既定）の場合は入力をそのまま返す。
    重みには**締切前オッズのみ**を使い、確定払戻（結果）は使わない。

    Args:
        config: 設定辞書
        creator: ModelCreator インスタンス
        sample_weight: 時間減衰サンプルウェイト
        train_meta: 学習行のメタ DataFrame（締切前オッズ列を含む）
        bet_type: 'win' / 'place' / 'umaren'

    Returns:
        np.ndarray: 合成後のサンプルウェイト
    """
    ow_cfg = config.get('model', {}).get('odds_weight', {})
    if not ow_cfg.get('enabled', False):
        return sample_weight

    # 馬券種ごとに gamma 等を上書きできるようにする（馬連はオッズレンジが広い）
    type_cfg = {**ow_cfg, **ow_cfg.get(bet_type, {})}
    gamma = float(type_cfg.get('gamma', 0.5))
    w_min = float(type_cfg.get('w_min', 0.5))
    w_max = float(type_cfg.get('w_max', 5.0))

    bet_col = BET_ODDS_COLS[bet_type]
    if bet_col not in train_meta.columns:
        logger.warning(
            f"[{bet_type}] 締切前オッズ列 '{bet_col}' が無いためオッズ重みをスキップします"
        )
        return sample_weight
    if len(train_meta) != len(sample_weight):
        raise ValueError(
            f"[{bet_type}] オッズ重みの行数が一致しません: "
            f"train_meta={len(train_meta)} != sample_weight={len(sample_weight)}"
        )

    odds = train_meta[bet_col].to_numpy(dtype=float)
    n_missing = int((~np.isfinite(odds)).sum())
    if n_missing > 0:
        logger.warning(
            f"[{bet_type}] 締切前オッズ欠損の {n_missing}/{len(odds)} 行は"
            "オッズ重み 1.0（等重み）で学習します"
        )
    odds_w = creator.compute_odds_weights(odds, gamma=gamma, w_min=w_min, w_max=w_max)
    logger.info(
        f"[{bet_type}] オッズ重みを適用: gamma={gamma} w_min={w_min} w_max={w_max} "
        f"（平均 {float(odds_w.mean()):.3f}）"
    )
    return (sample_weight * odds_w).astype(np.float32)


def _attach_ranking_rank(
    creator: 'ModelCreator',
    eval_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """評価用 DataFrame にレース内のランキング順位 ``ranking_rank`` を付与する。

    ``evaluation.min_ranking_rank`` は ``ranking_rank`` 列が存在するときのみ
    適用される（``ModelEvaluator.calculate_recovery_rate``）。この列が無いと
    設定値が黙って無視され、学習時評価と当日予測（``predictor``）とで
    賭け条件が食い違うため、ここで明示的に生成する。

    ランキングモデルはバイナリ（win/place）より後に学習されるため、学習中の
    モデルは使えない。**前回の学習で保存済みの ``ranking_model.txt``** を読み込む
    （＝評価対象期間を学習していない別アーティファクトなのでリークしない。
    当日予測が使うモデルと同一という点でも本番条件に一致する）。

    Args:
        creator: ModelCreator インスタンス
        eval_df: 評価対象 DataFrame（``race_id`` 必須）
        feature_cols: モデル入力特徴量カラム

    Returns:
        pd.DataFrame: ``ranking_score`` / ``ranking_rank`` を付与した DataFrame。
            ランキングモデルが利用できない場合は入力をそのまま返す。
    """
    min_ranking_rank = creator.min_ranking_rank
    if min_ranking_rank <= 0:
        return eval_df

    if 'ranking' not in creator.models:
        model_path = creator.model_dir / 'ranking_model.txt'
        if not model_path.exists():
            logger.warning(
                f"evaluation.min_ranking_rank={min_ranking_rank} が設定されていますが、"
                f"ランキングモデル（{model_path}）が見つからないため "
                "ランキング順位フィルターを適用せずに評価します。"
                "--model-type all で一度学習すると次回以降適用されます。"
            )
            return eval_df
        creator.load_model('ranking_model.txt', 'ranking')
        logger.info(
            f"ランキング順位フィルター用に保存済みモデルを読み込みました: {model_path}"
        )

    # 保存済みランキングモデルが現在の特徴量スキーマと食い違う場合はスキップする。
    # 特徴量を追加・削除した直後の学習では、前回保存のモデルは旧スキーマのままで
    # あり、そのまま predict すると LightGBMError で学習全体が落ちる。
    # ランキングモデルはこの後の工程で新スキーマで学習・保存されるため、
    # 次回以降の学習では正しく適用される。
    ranking_model = creator.models.get('ranking')
    try:
        expected_n = ranking_model.num_feature() if ranking_model is not None else None
    except AttributeError:
        # num_feature を持たないモデル実装（テストのスタブ等）では検査を省略する
        expected_n = None
    if expected_n is not None and expected_n != len(feature_cols):
        logger.warning(
            f"保存済みランキングモデルの特徴量数（{expected_n}）が現在のスキーマ"
            f"（{len(feature_cols)}）と一致しないため、ランキング順位フィルターを"
            "適用せずに評価します。特徴量セットを変更した直後の学習では想定内です"
            "（ランキングモデルはこの後の工程で新スキーマで再学習・保存されるため、"
            "次回以降の学習では適用されます）。"
        )
        creator.models.pop('ranking', None)
        return eval_df

    out = eval_df.copy()
    out['ranking_score'] = creator.predict(out[list(feature_cols)], model_name='ranking')
    out['ranking_rank'] = (
        out.groupby('race_id')['ranking_score']
        .rank(ascending=False, method='min')
        .astype(int)
    )
    logger.info(
        f"ranking_rank を付与しました（min_ranking_rank={min_ranking_rank} で "
        f"{int((out['ranking_rank'] <= min_ranking_rank).sum())}/{len(out)} 行が賭け対象候補）"
    )
    return out


def _train_and_evaluate_model(
    config: dict, config_path: str, creator: 'ModelCreator',
    df: pd.DataFrame, feature_cols: List[str], model_name: str, target_col: str,
    use_optuna: bool,
) -> Tuple[Dict, Dict]:
    """単一バイナリモデルの学習・評価・回収率計算を行う（win/place 共通）。"""
    logger.info(f"=== {_LABEL_MAP.get(model_name, model_name)}モデル学習 ===")
    X_train, X_test, y_train, y_test, train_dates = creator.prepare_data(df, feature_cols, target_col)
    decay_cfg = config.get('model', {}).get('time_decay', {})
    sample_weight = creator.compute_time_decay_weights(
        train_dates,
        recent_years=float(decay_cfg.get('recent_years', 2.0)),
        mid_years=float(decay_cfg.get('mid_years', 4.0)),
        w_recent=float(decay_cfg.get('w_recent', 3.0)),
        w_mid=float(decay_cfg.get('w_mid', 2.0)),
    )

    # 学習/テストのメタ行を prepare_data と同じ分割日で切り出す
    split_date = LightGBMTrainer.compute_split_date(df)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    train_meta = df_sorted[df_sorted['date'] < split_date]
    test_meta = df_sorted[df_sorted['date'] >= split_date].copy()

    # Phase 2a: オッズ重み（高オッズ的中を重視）を時間減衰ウェイトに掛け合わせる。
    # 重みには締切前オッズのみを使う（確定払戻を使うとラベルリーク）。
    sample_weight = _apply_odds_weights(config, creator, sample_weight, train_meta, model_name)

    best_params: Dict = {}
    if use_optuna:
        n_trials = config.get('model', {}).get('optuna_trials', 100)
        # 賭け判断＝締切前オッズ / 払戻＝確定払戻金と分離済みなので recovery_rate を解禁する。
        # 2系統オッズが揃っていない場合のみ auc にフォールバックする。
        hpo_objective = config.get('model', {}).get('hpo_objective', 'auc')
        val_meta_df: Optional[pd.DataFrame] = None
        if hpo_objective == 'recovery_rate':
            if has_two_source_odds(test_meta, model_name):
                val_meta_df = test_meta
                logger.info(
                    f"[{model_name}] hpo_objective='recovery_rate' で最適化します"
                    "（賭け判断=締切前オッズ / 払戻=確定払戻金）。"
                )
            else:
                logger.warning(
                    f"[{model_name}] 締切前オッズ・確定払戻列が揃っていないため "
                    "hpo_objective を 'auc' にフォールバックします。"
                )
                hpo_objective = 'auc'
        # Phase 2b: fobj_beta_search が有効なら、HPO の各 trial で beta も同時に探索する。
        # sample_weight（オッズ重み×時間減衰）は最終学習と同じものを使う。
        fobj_beta_range = _resolve_fobj_beta_search_range(config, model_name)
        best_params = creator.hyperparameter_tuning(
            X_train, y_train, X_test, y_test, n_trials=n_trials, model_name=model_name,
            val_meta_df=val_meta_df, objective_metric=hpo_objective,
            fobj_sample_weight=sample_weight if fobj_beta_range is not None else None,
            fobj_beta_range=fobj_beta_range,
        )
        if best_params:
            _section_key = 'lightgbm_place' if model_name == 'place' else 'lightgbm'
            # fobj_beta は LightGBM のハイパーパラメータではないため、
            # lightgbm セクションには書き戻さず個別に扱う（tunable keys の対象外）。
            lgbm_params = {k: v for k, v in best_params.items() if k != 'fobj_beta'}
            creator.update_lgbm_params(_section_key, lgbm_params)
            _write_lgbm_params_to_config(config, config_path, lgbm_params, model_name)

    # 回収率が最良のイテレーションを記録する（use_profit_iter=True で採用可能）
    val_df_for_profit = test_meta if has_two_source_odds(test_meta, model_name) else None
    if use_optuna and 'fobj_beta' in best_params:
        # HPO で探索した beta を最終学習に採用する
        fobj_beta: Optional[float] = float(best_params['fobj_beta'])
        logger.info(f"[{model_name}] HPO で探索した fobj_beta={fobj_beta:.3f} を最終学習に採用します")
    else:
        fobj_beta = _resolve_fobj_beta(config, model_name)
    creator.train(
        X_train, y_train, X_test, y_test, model_name=model_name,
        sample_weight=sample_weight, val_df_for_profit=val_df_for_profit,
        fobj_beta=fobj_beta,
    )

    _fit_calibrator(
        config, creator, model_name, df, split_date, feature_cols, target_col
    )

    # 評価テストセットは prepare_data と同一の分割日で切り出す（test_meta と同一）。
    # Timedelta(days=180) だと月の日数次第で分割がずれ、学習データが評価に混入する。
    # ranking_rank は行を並べ替えないため y_pred_test との行対応は保たれる。
    test_df = _attach_ranking_rank(creator, test_meta, feature_cols)
    y_pred_test = creator.predict(test_df[list(feature_cols)], model_name=model_name)
    # metrics と recovery は同一（キャリブレーション後）の確率で計算し整合させる
    metrics = creator.evaluate_model(test_df[target_col], y_pred_test, threshold=0.5)
    min_ev = config.get('evaluation', {}).get(f'min_ev_{model_name}', 1.05)
    recovery = creator.calculate_recovery_rate(test_df, y_pred_test, min_ev=min_ev, bet_type=model_name)
    creator.print_summary(metrics, recovery)
    return metrics, recovery


def _train_and_evaluate_umaren(
    config: dict,
    config_path: str,
    creator: 'ModelCreator',
    df: pd.DataFrame,
    feature_cols: List[str],
    use_optuna: bool,
) -> Tuple[Dict, Dict]:
    """馬連（umaren）モデルの学習・評価・回収率計算を行う。

    馬単位の特徴量をペア単位（1行1組み合わせ）に展開し、win/place と同じ
    binary 分類 → EV フィルタ → Kelly → 回収率最大化の枠組みで学習する。

    Args:
        config: 設定辞書
        config_path: config.yaml のパス（Optuna 結果の書き戻し先）
        creator: ModelCreator インスタンス
        df: 馬単位の特徴量 DataFrame（メタ列結合済み）
        feature_cols: 馬単位のモデル特徴量カラム
        use_optuna: ハイパーパラメータ最適化を行うか

    Returns:
        Tuple[Dict, Dict]: (評価指標, 回収率結果)

    Raises:
        ValueError: 単勝モデルが未学習の場合（候補絞り込みに必要）
    """
    from src.models.pair_builder import (
        add_umaren_labels,
        build_umaren_pairs,
        pair_feature_cols,
    )

    logger.info("=== 馬連モデル学習 ===")

    umaren_cfg = config.get('model', {}).get('umaren', {})
    top_k = int(umaren_cfg.get('top_k_horses', 8))
    aggregations = tuple(umaren_cfg.get('pair_aggregations', ['max', 'min', 'sum', 'absdiff']))

    # 候補絞り込みに使う単勝確率は、当該レースを学習に含まないモデルの出力でなければ
    # ならない。ここでは temporal split の学習期間で学習済みの win モデルを使い、
    # 学習期間の行には in-sample 確率が出るが、絞り込み（＝候補集合の決定）にのみ
    # 使い、特徴量としては pair_win_proba_* に閉じる。
    if 'win' not in creator.models:
        raise ValueError(
            "馬連モデルの候補絞り込みには学習済みの単勝モデルが必要です。"
            "--model-type all もしくは enabled_models に 'win' を含めて実行してください。"
        )
    win_proba = creator.predict(df[list(feature_cols)], model_name='win')

    pairs = build_umaren_pairs(
        df, feature_cols, win_proba=win_proba,
        top_k_horses=top_k, aggregations=aggregations,
    )
    payouts = load_payouts(config.get('data', {}).get('payouts_dir', 'data/processed/payouts'))
    pairs = add_umaren_labels(pairs, build_umaren_payout(payouts))
    pairs = attach_umaren_meta_columns(pairs, config)

    pair_cols = pair_feature_cols(pairs)
    logger.info(f"馬連ペア特徴量: {len(pair_cols)} 個")

    X_train, X_test, y_train, y_test, train_dates = creator.prepare_data(
        pairs, pair_cols, 'target_umaren'
    )
    decay_cfg = config.get('model', {}).get('time_decay', {})
    sample_weight = creator.compute_time_decay_weights(
        train_dates,
        recent_years=float(decay_cfg.get('recent_years', 2.0)),
        mid_years=float(decay_cfg.get('mid_years', 4.0)),
        w_recent=float(decay_cfg.get('w_recent', 3.0)),
        w_mid=float(decay_cfg.get('w_mid', 2.0)),
    )

    split_date = LightGBMTrainer.compute_split_date(pairs)
    pairs_sorted = pairs.sort_values('date').reset_index(drop=True)
    train_meta = pairs_sorted[pairs_sorted['date'] < split_date]
    test_meta = pairs_sorted[pairs_sorted['date'] >= split_date].copy()

    sample_weight = _apply_odds_weights(config, creator, sample_weight, train_meta, 'umaren')

    best_params: Dict = {}
    if use_optuna:
        n_trials = config.get('model', {}).get('optuna_trials', 100)
        hpo_objective = config.get('model', {}).get('hpo_objective', 'auc')
        val_meta_df: Optional[pd.DataFrame] = None
        if hpo_objective == 'recovery_rate':
            if has_two_source_odds(test_meta, 'umaren'):
                val_meta_df = test_meta
            else:
                logger.warning(
                    "[umaren] 締切前オッズ・確定払戻列が揃っていないため "
                    "hpo_objective を 'auc' にフォールバックします。"
                )
                hpo_objective = 'auc'
        fobj_beta_range = _resolve_fobj_beta_search_range(config, 'umaren')
        best_params = creator.hyperparameter_tuning(
            X_train, y_train, X_test, y_test, n_trials=n_trials, model_name='umaren',
            val_meta_df=val_meta_df, objective_metric=hpo_objective,
            fobj_sample_weight=sample_weight if fobj_beta_range is not None else None,
            fobj_beta_range=fobj_beta_range,
        )
        if best_params:
            lgbm_params = {k: v for k, v in best_params.items() if k != 'fobj_beta'}
            creator.update_lgbm_params('lightgbm_umaren', lgbm_params)
            _write_lgbm_params_to_config(config, config_path, lgbm_params, 'umaren')

    val_df_for_profit = test_meta if has_two_source_odds(test_meta, 'umaren') else None
    if use_optuna and 'fobj_beta' in best_params:
        fobj_beta: Optional[float] = float(best_params['fobj_beta'])
        logger.info(f"[umaren] HPO で探索した fobj_beta={fobj_beta:.3f} を最終学習に採用します")
    else:
        fobj_beta = _resolve_fobj_beta(config, 'umaren')
    creator.train(
        X_train, y_train, X_test, y_test, model_name='umaren',
        sample_weight=sample_weight, val_df_for_profit=val_df_for_profit,
        fobj_beta=fobj_beta,
    )

    _fit_calibrator(
        config, creator, 'umaren', pairs, split_date, pair_cols, 'target_umaren'
    )

    y_pred_test = creator.predict(test_meta[pair_cols], model_name='umaren')
    metrics = creator.evaluate_model(test_meta['target_umaren'], y_pred_test, threshold=0.5)
    min_ev = config.get('evaluation', {}).get('min_ev_umaren', 1.05)
    recovery = creator.calculate_recovery_rate(
        test_meta, y_pred_test, min_ev=min_ev, bet_type='umaren'
    )
    creator.print_summary(metrics, recovery)
    return metrics, recovery


def _save_binary_models(config: dict, creator: 'ModelCreator') -> None:
    """enabled_models に含まれるバイナリモデルをファイルに保存し、重要度をログ出力する。"""
    enabled = config.get('model', {}).get('enabled_models', ['win', 'place'])
    save_map = {
        'win': 'win_model.txt',
        'place': 'place_model.txt',
    }
    for model_name, filename in save_map.items():
        if model_name in enabled:
            creator.save_model(model_name, filename)
    if 'win' in creator.models:
        importance = creator.get_feature_importance('win', top_n=20)
        logger.info(f"\n単勝モデル 特徴量重要度:\n{importance}")


def _evaluate_ranking_model(
    creator: 'ModelCreator', test_df: pd.DataFrame, X_test: pd.DataFrame, groups_test: List[int]
) -> Dict:
    """ランキングモデルの簡易評価（top-1的中率）。

    profit callback で回収率最良イテレーションが記録されている場合は、
    NDCG最良（best_iteration）版と回収率最良版の top-1的中率を test で比較ログ出力する
    （どちらを本番採用すべきかの判断材料。valid で選んだ profit iter を test で検証する形で
    リークにはならない）。
    """
    if 'finish_position' not in test_df.columns:
        # 欠損を黙って的中0扱いにせず、想定外として明示的に止める
        raise ValueError(
            "finish_position カラムがないため top-1 的中率を計算できません"
        )

    def _top1_hit_rate(scores: np.ndarray) -> Tuple[float, int, int]:
        eval_df = test_df.copy()
        eval_df['ranking_score'] = scores
        hit_count = 0
        race_count = 0
        idx_start = 0
        for group_size in groups_test:
            group_df = eval_df.iloc[idx_start: idx_start + group_size]
            idx_start += group_size
            top_horse = group_df.loc[group_df['ranking_score'].idxmax()]
            if int(top_horse['finish_position']) == 1:
                hit_count += 1
            race_count += 1
        rate = hit_count / race_count * 100 if race_count > 0 else 0.0
        return rate, hit_count, race_count

    scores = creator.predict(X_test, model_name='ranking')
    hit_rate, hit_count, race_count = _top1_hit_rate(scores)
    logger.info(f"[ranking] top-1的中率(NDCG最良): {hit_rate:.2f}%  ({hit_count}/{race_count} レース)")

    result: Dict = {
        'top1_hit_rate': hit_rate, 'hit_count': hit_count, 'race_count': race_count,
    }

    # 回収率最良イテレーションが記録されていれば、その版の top-1的中率も出す
    profit_iter = creator.profit_iterations.get('ranking')
    if profit_iter is not None:
        scores_profit = creator.predict(X_test, model_name='ranking', use_profit_iter=True)
        pr_rate, pr_hit, pr_race = _top1_hit_rate(scores_profit)
        logger.info(
            f"[ranking] top-1的中率(回収率最良iter={profit_iter}): "
            f"{pr_rate:.2f}%  ({pr_hit}/{pr_race} レース)"
        )
        result['top1_hit_rate_profit_iter'] = pr_rate
        result['profit_iteration'] = profit_iter

    return result


def _train_ranking(
    config: dict, config_path: str, feature_file: Optional[str], use_optuna: bool,
    df: Optional[pd.DataFrame] = None, feature_cols: Optional[List[str]] = None,
) -> Optional[Dict]:
    """ランキングモデル学習サブルーチン。

    df / feature_cols が渡された場合はロード済みとして再利用し、
    特徴量ファイルの二重読み込み・model_feature.json の二重保存を回避する。
    """
    logger.info("=== ランキングモデル学習 ===")
    if df is None or feature_cols is None:
        df, feature_cols = _load_features_for_training(config, feature_file)

    if 'finish_position' not in df.columns:
        logger.error(
            "finish_position カラムが見つかりません。ランキング学習には必須です。"
        )
        return None

    ranking_cv: Dict = {}
    cv_config = config.get('model', {}).get('cross_validation', {})
    if cv_config.get('enabled', False):
        creator_cv = ModelCreator(config, config.get('features_def'))
        ranking_cv = _run_cv_evaluation_ranking(config, creator_cv, df, feature_cols, cv_config)

    ranking_label_config = config.get('model', {}).get('ranking_label', {
        'top1': 2, 'top3': 1, 'other': 0
    })
    creator = ModelCreator(config, config.get('features_def'))
    (X_train, X_test, y_train, y_test,
     groups_train, groups_test, train_df, test_df,
     ranking_feature_cols) = creator.prepare_ranking_data(df, feature_cols, ranking_label_config)

    # early stopping / profit callback 用の検証セットを test から分離する（楽観バイアス防止）。
    # 学習期間（train_df）の末尾 valid_window_months ヶ月を valid ホールドアウトに切り出し、
    # fit にはそれ以前を使う。valid を確保できない場合のみ従来どおり test を使う。
    valid_months = int(cv_config.get('valid_window_months', 2))
    es_split_date = train_df['date'].max() - pd.DateOffset(months=valid_months)
    fit_mask = train_df['date'] < es_split_date
    val_mask = ~fit_mask
    if fit_mask.any() and val_mask.any():
        fit_df_r = train_df[fit_mask].reset_index(drop=True)
        val_df_r = train_df[val_mask].reset_index(drop=True)
        X_fit = fit_df_r[ranking_feature_cols]
        y_fit = y_train[fit_mask.to_numpy()].reset_index(drop=True)
        groups_fit = fit_df_r.groupby('race_id', sort=False).size().tolist()
        X_val_es = val_df_r[ranking_feature_cols]
        y_val_es = y_train[val_mask.to_numpy()].reset_index(drop=True)
        groups_val_es = val_df_r.groupby('race_id', sort=False).size().tolist()
        profit_df = val_df_r
    else:
        # フォールバック: 学習期間が短く valid を取れないため test を使う
        logger.warning(
            "ランキング学習: 学習期間が短く valid ホールドアウトを確保できないため、"
            "early stopping/profit 観測に test を使います（回収率が楽観的になる可能性があります）"
        )
        X_fit, y_fit, groups_fit = X_train, y_train, groups_train
        X_val_es, y_val_es, groups_val_es = X_test, y_test, groups_test
        profit_df = test_df

    if use_optuna:
        n_trials = config.get('model', {}).get('optuna_trials', 100)
        best_params = creator.hyperparameter_tuning_ranking(
            X_fit, y_fit, groups_fit, X_val_es, y_val_es, groups_val_es, n_trials=n_trials,
        )
        if best_params:
            creator.update_lgbm_params('lightgbm_ranking', best_params)
            _write_lgbm_params_to_config(config, config_path, best_params, 'ranking')
    creator.train_ranking(
        X_fit, y_fit, groups_fit, X_val_es, y_val_es, groups_val_es,
        model_name='ranking', val_df_for_profit=profit_df,
        val_feature_cols_for_profit=ranking_feature_cols,
    )
    creator.save_model('ranking', 'ranking_model.txt')
    logger.info("ランキングモデル保存完了: ranking_model.txt")

    importance = creator.get_feature_importance('ranking', top_n=20)
    logger.info(f"\nランキングモデル 特徴量重要度:\n{importance}")

    ranking_eval = _evaluate_ranking_model(creator, test_df, X_test, groups_test)
    result: Dict = {'importance': importance, **ranking_eval}
    if ranking_cv:
        result['cv'] = ranking_cv
    return result


def _run_blend_alpha_optimization(
    config: dict, config_path: str, df: pd.DataFrame, feature_cols: List[str]
) -> None:
    """Optunaで blend_alpha / ranking_temperature を最適化し、config.yaml に書き戻す。"""
    logger.info("=== モデルブレンド最適化開始 ===")
    blend_opt_cfg = config.get('evaluation', {}).get('blend_optimization', {})
    cv_config = config.get('model', {}).get('cross_validation', {})
    ranking_label_config = config.get('model', {}).get('ranking_label', {
        'top1': 2, 'top3': 1, 'other': 0
    })
    n_splits = blend_opt_cfg.get('n_splits', cv_config.get('n_splits', 3))
    n_trials = blend_opt_cfg.get('n_trials', 30)
    eval_cfg = config.get('evaluation', {})
    min_ev_win = eval_cfg.get('min_ev_win', 1.05)
    min_ev_place = eval_cfg.get('min_ev_place', 1.05)
    min_train_months = cv_config.get('min_train_months', 12)
    test_window_months = cv_config.get('test_window_months', 3)
    cv_type = cv_config.get('cv_type', 'expanding')
    if cv_type == 'both':
        cv_type = 'expanding'

    creator = ModelCreator(config, config.get('features_def'))
    result = creator.optimize_blend_alpha_cv(
        df=df, feature_cols=feature_cols, ranking_label_config=ranking_label_config,
        min_ev_win=min_ev_win, min_ev_place=min_ev_place, n_splits=n_splits,
        min_train_months=min_train_months, test_window_months=test_window_months,
        cv_type=cv_type, n_trials=n_trials,
    )
    if not result:
        logger.warning("blend最適化が結果を返しませんでした。設定は更新しません。")
        return

    best_alpha: float = result['best_alpha']
    best_alpha_place: float = result['best_alpha_place']
    best_temperature: float = result['best_temperature']
    best_rr: float = result['best_recovery_rate']
    prev_alpha = eval_cfg.get('blend_alpha', 0.6)
    prev_alpha_place = eval_cfg.get('blend_alpha_place', prev_alpha)
    prev_temp = eval_cfg.get('ranking_temperature', 1.5)
    logger.info(
        f"blend_alpha:          {prev_alpha:.3f} → {best_alpha:.3f}\n"
        f"blend_alpha_place:    {prev_alpha_place:.3f} → {best_alpha_place:.3f}\n"
        f"ranking_temperature:  {prev_temp:.3f} → {best_temperature:.3f}\n"
        f"CV回収率:              {best_rr:.1f}%"
    )
    _update_config_yaml(config, config_path, {
        'evaluation.blend_alpha': round(best_alpha, 4),
        'evaluation.blend_alpha_place': round(best_alpha_place, 4),
        'evaluation.ranking_temperature': round(best_temperature, 4),
    })


def run_train(
    config: dict,
    model_type: str = 'all',
    feature_file: Optional[str] = None,
    config_path: str = 'config/config.yaml',
) -> Dict:
    """モデル学習を実行する。

    --model-type に応じてバイナリ（win/place）・ランキング・馬連・全部を学習し、
    学習済みモデルと model_feature.json を保存する。
    デフォルトは 'all'（win / place / ranking / umaren の4モデルを生成。
    ただし umaren は enabled_models に含まれる場合のみ）。

    Args:
        config: 設定辞書（config.yaml + features_def をロード済み）
        model_type: 'binary' / 'ranking' / 'umaren' / 'all'
        feature_file: 特徴量ファイル名（省略時は 'features.feather'）
        config_path: config.yaml のパス（Optuna 最適化結果の書き戻し先）

    Returns:
        Dict: レポート用に収集した評価データ辞書。
    """
    logger.info("モデル学習モード開始（model_type=%s）", model_type)
    use_optuna = config.get('model', {}).get('use_optuna', False)

    df: Optional[pd.DataFrame] = None
    feature_cols: Optional[List[str]] = None
    creator: Optional[ModelCreator] = None
    report_data: Dict = {}

    # 馬連は単勝モデルの予測で候補を絞るため binary の学習が前提になる
    need_binary = model_type in ('binary', 'all', 'umaren')

    if need_binary:
        df, feature_cols = _load_features_for_training(config, feature_file)
        creator = ModelCreator(config, config.get('features_def'))

        test_cutoff = df['date'].max() - pd.Timedelta(days=180)
        report_data['train_period'] = (df['date'].min(), test_cutoff)
        report_data['test_period'] = (test_cutoff, df['date'].max())

        cv_config = config.get('model', {}).get('cross_validation', {})
        if cv_config.get('enabled', False):
            report_data['cv_binary'] = _run_cv_evaluation(config, creator, df, feature_cols, cv_config)

        enabled = list(config.get('model', {}).get('enabled_models', ['win', 'place']))
        if model_type == 'umaren' and 'win' not in enabled:
            # 馬連の候補絞り込みに単勝モデルが必須なので強制的に学習する
            logger.info("--model-type umaren のため単勝モデルを学習します（候補絞り込みに必要）")
            enabled.append('win')
        for model_name, target_col in [('win', 'target_win'), ('place', 'target_place')]:
            if model_name not in enabled:
                logger.info(f"{model_name} モデルはスキップ（enabled_models に含まれていない）")
                continue
            if model_type == 'umaren' and model_name == 'place':
                continue  # 馬連単独実行では複勝は不要
            metrics, recovery = _train_and_evaluate_model(
                config, config_path, creator, df, feature_cols, model_name, target_col, use_optuna
            )
            report_data[model_name] = {'metrics': metrics, 'recovery': recovery}
        if model_type != 'umaren':
            _save_binary_models(config, creator)

        for mn in ['win', 'place']:
            if mn in creator.models and mn in report_data:
                report_data[mn]['importance'] = creator.get_feature_importance(mn, top_n=20)

    # 馬連モデル（'all' では enabled_models に含まれる場合のみ、'umaren' 指定時は常に学習）
    enabled_models = config.get('model', {}).get('enabled_models', ['win', 'place'])
    train_umaren = model_type == 'umaren' or (
        model_type == 'all' and 'umaren' in enabled_models
    )
    if train_umaren and creator is not None and df is not None and feature_cols is not None:
        metrics, recovery = _train_and_evaluate_umaren(
            config, config_path, creator, df, feature_cols, use_optuna
        )
        report_data['umaren'] = {
            'metrics': metrics,
            'recovery': recovery,
            'importance': creator.get_feature_importance('umaren', top_n=20),
        }
        creator.save_model('umaren', 'umaren_model.txt')
        logger.info("馬連モデル保存完了: umaren_model.txt")

    if model_type in ('ranking', 'all'):
        # 'all' 時は binary で読み込んだ df/feature_cols を再利用し二重ロードを避ける
        ranking_result = _train_ranking(
            config, config_path, feature_file, use_optuna, df=df, feature_cols=feature_cols
        )
        if ranking_result:
            report_data['ranking'] = ranking_result

    # blend_alpha / ranking_temperature の Optuna 最適化（model_type=all のみ）
    blend_opt_cfg = config.get('evaluation', {}).get('blend_optimization', {})
    if model_type == 'all' and blend_opt_cfg.get('enabled', False) and df is not None and feature_cols is not None:
        _run_blend_alpha_optimization(config, config_path, df, feature_cols)

    logger.info("モデル学習モード完了")
    return report_data


def run_train_stack(
    config: dict,
    feature_file: Optional[str] = None,
) -> Dict:
    """多専門家スタッキングを学習する（単独実行ルーチン）。

    ``docs/report/20260811_market_edge_analysis.md`` §5 P1 の
    「市場アンカー型の残差学習」を Meta Model として構成する。

    処理の流れ:
        1. 特徴量をロードし、専門家ごとの入力列を解決・保存する
        2. walk-forward で全専門家の OOF 予測を生成する
        3. OOF から Meta Model（条件付きロジット）を学習する
        4. 全データで各専門家を再学習し、Meta とともに保存する

    Args:
        config: 全体設定辞書
        feature_file: 特徴量ファイル名

    Returns:
        Dict: 学習結果のサマリ

    Raises:
        ValueError: 専門家の設定が不正、または OOF が生成できない場合
    """
    from src.cli_common import derive_expert_feature_cols, save_expert_feature_schema
    from src.models.experts import resolve_roster
    from src.models.oof import generate_oof_predictions
    from src.models.stacking import (
        build_stack_oof_frame,
        save_stack,
        train_stack_meta,
    )

    logger.info("=== 多専門家スタッキング学習開始 ===")

    ens_cfg = config.get('model', {}).get('ensemble', {})
    expert_ids = ens_cfg.get('experts', ['e1_ability', 'e2_ranking', 'e5a_market'])
    experts = resolve_roster(list(expert_ids))

    df, _ = _load_features_for_training(config, feature_file)
    features_def = config.get('features_def') or {}

    expert_cols: Dict[str, List[str]] = {}
    for spec in experts:
        cols = derive_expert_feature_cols(
            df.columns.tolist(), features_def, spec.feature_group
        )
        for extra in spec.extra_feature_groups:
            extra_cols = derive_expert_feature_cols(
                df.columns.tolist(), features_def, extra
            )
            cols = list(dict.fromkeys(cols + extra_cols))
        expert_cols[spec.expert_id] = cols
        path = save_expert_feature_schema(config, spec.expert_id, cols)
        logger.info(
            "[%s] 入力特徴量 %d 列 / スキーマ保存: %s",
            spec.expert_id, len(cols), path,
        )

    creator = ModelCreator(config, features_def)
    oof_cfg = ens_cfg.get('oof', {})
    ranking_label_config = config.get('model', {}).get('ranking_label', {
        'top1': 2, 'top3': 1, 'other': 0
    })

    fold_data = generate_oof_predictions(
        creator._optimizer, df, experts, expert_cols, ranking_label_config,
        n_splits=int(oof_cfg.get('n_splits', 5)),
        min_train_months=int(oof_cfg.get('min_train_months', 12)),
        test_window_months=int(oof_cfg.get('test_window_months', 3)),
        cv_type=str(oof_cfg.get('cv_type', 'expanding')),
    )

    oof_df = build_stack_oof_frame(fold_data, [e.expert_id for e in experts])
    meta_cfg = ens_cfg.get('meta', {})
    c_value = meta_cfg.get('C')
    c_candidates = (float(c_value),) if c_value is not None else (0.01, 0.1, 1.0)

    meta_model, meta_cols, info = train_stack_meta(
        oof_df, [e.expert_id for e in experts], c_candidates=c_candidates
    )

    model_dir = Path(config['data'].get('model_dir', 'data/model'))
    save_stack(model_dir, meta_model, meta_cols, [e.expert_id for e in experts], info)

    # 最終モデルの学習からはテスト期間を除外する。win/place/ranking と同じ
    # split_date を使い、保存済みモデルが「テスト期間を学習していない」状態を
    # 揃える（揃えないとホールドアウト評価が in-sample になり AUC=1.0 になる）。
    split_date = LightGBMTrainer.compute_split_date(df)
    train_df = df[df['date'] < split_date].reset_index(drop=True)
    logger.info(
        "最終モデルの学習期間: 〜%s（%s 行）／テスト期間 %s 行は除外",
        split_date.date(), f'{len(train_df):,}', f'{len(df) - len(train_df):,}',
    )
    _train_final_experts(creator, train_df, experts, expert_cols, ranking_label_config)

    logger.info("=== 多専門家スタッキング学習完了 ===")
    return {'meta': info, 'expert_cols': {k: len(v) for k, v in expert_cols.items()}}


def _train_final_experts(
    creator: 'ModelCreator',
    df: pd.DataFrame,
    experts: List,
    expert_cols: Dict[str, List[str]],
    ranking_label_config: dict,
) -> None:
    """学習期間の全データで各専門家を再学習し ``data/model/stack/`` に保存する。

    OOF はフォールドごとの部分データで学習したモデルによるものなので、
    本番推論用には学習期間をまとめて使い直したモデルを保存する。

    IMPORTANT:
        ``df`` には**テスト期間を含めてはならない**。含めるとホールドアウト評価が
        in-sample になり AUC=1.0・ROI 1000% 超という無意味な数字が出る
        （2026-08-22 の Stage 2 初回評価で実際に発生）。
        呼び出し側で ``compute_split_date`` により分割済みの DataFrame を渡すこと。
    """
    from src.models.oof import _apply_min_date
    from src.models.ranking_label import make_relevance_labels

    for spec in experts:
        cols = expert_cols[spec.expert_id]
        fit_df = _apply_min_date(df, spec)
        if len(fit_df) == 0:
            logger.warning(
                "[%s] min_date=%s により学習データが0件のためスキップします",
                spec.expert_id, spec.min_date,
            )
            continue

        init_score = None
        if spec.use_market_init_score:
            from src.features.market_probability import compute_market_proba, safe_logit
            from src.models.odds_series import BET_ODDS_COLS

            odds = fit_df[BET_ODDS_COLS['win']].to_numpy(dtype=float)
            valid = np.isfinite(odds) & (odds > 0)
            fit_df = fit_df[valid].reset_index(drop=True)
            init_score = safe_logit(
                compute_market_proba(
                    fit_df[BET_ODDS_COLS['win']].to_numpy(dtype=float),
                    fit_df['race_id'], validate_overround=False,
                )
            )

        if spec.objective == 'lambdarank':
            groups = fit_df.groupby('race_id', sort=False).size().tolist()
            y = pd.Series(
                make_relevance_labels(fit_df, ranking_label_config, None),
                index=fit_df.index,
            )
            creator._trainer.train_ranking(
                fit_df[cols], y, groups, model_name=spec.expert_id
            )
        else:
            creator._trainer.train(
                fit_df[cols], fit_df[spec.target_col],
                model_name=spec.expert_id, init_score=init_score,
            )

        creator._trainer.save_model(
            spec.expert_id, str(Path('stack') / f'{spec.expert_id}.txt')
        )
        logger.info("[%s] 最終モデルを保存しました", spec.expert_id)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='モデル生成（単独実行）')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='設定ファイルパス')
    parser.add_argument('--feature-file', type=str, help='特徴量ファイル名（省略時は features.feather）')
    parser.add_argument(
        '--model-type', type=str, default='all',
        choices=['binary', 'ranking', 'umaren', 'all', 'stack'],
        help=(
            '学習モデルタイプ: binary=win/place, ranking=ランキング, umaren=馬連, '
            'all=全モデル（デフォルト。umaren は enabled_models に含まれる場合のみ）, '
            'stack=多専門家スタッキング（専門家 + Meta Model）'
        )
    )
    return parser.parse_args()


def main() -> None:
    from src.cli_common import build_config

    args = _parse_args()
    config = build_config(args.config)
    if args.model_type == 'stack':
        run_train_stack(config, feature_file=args.feature_file)
        return
    run_train(
        config,
        model_type=args.model_type,
        feature_file=args.feature_file,
        config_path=args.config,
    )


if __name__ == '__main__':
    main()
