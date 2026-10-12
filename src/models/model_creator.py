"""LightGBM 二値分類モデルの学習・確率校正・予測・保存・読込。

単勝モデル（:mod:`src.simulator.retrain` / :mod:`src.simulator.predict_today`）が使う。
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def _check_model_line_endings(model_path: Path) -> None:
    """LightGBM モデルファイルが CRLF になっていないか検証する。

    git の core.autocrlf で CRLF に変換されたモデルを LightGBM に読ませると、
    Python の例外ではなくプロセスごと異常終了する（exit 127）ため先に止める。

    Raises:
        ValueError: ファイルが CRLF を含む場合
    """
    with open(model_path, 'rb') as f:
        head = f.read(4096)
    if b'\r\n' not in head:
        return
    raise ValueError(
        f"モデルファイルの改行コードが CRLF です: {model_path}\n"
        "LightGBM は CRLF のモデルを読めず、プロセスごと異常終了します。\n"
        "対処: .gitattributes に `data/model/**/*.txt -text` を追加し、"
        "ファイルの改行を LF に戻してください。"
    )


class ModelCreator:
    """LightGBM 二値分類モデルと確率校正器を管理する。"""

    # isotonic 校正後の確率の下限。EV = 確率 × オッズ が恒久的に 0 になるのを防ぐ。
    CALIBRATED_PROBA_FLOOR = 1e-5

    def __init__(self, config: dict):
        """
        Args:
            config: 全体設定辞書（``model`` セクションを参照する）
        """
        self.config = config.get('model', {})
        self.model_dir = Path(self.config.get('model_dir', 'data/model'))
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, lgb.Booster] = {}
        self.calibrators: Dict[str, tuple] = {}

    @staticmethod
    def compute_time_decay_weights(
        dates: pd.Series,
        recent_years: float = 2.0,
        mid_years: float = 4.0,
        w_recent: float = 3.0,
        w_mid: float = 2.0,
    ) -> np.ndarray:
        """直近のレースほど重くする時間減衰サンプルウェイトを計算する。"""
        days = (dates.max() - dates).dt.days
        weights = np.ones(len(dates), dtype=np.float32)
        weights[days <= 365 * recent_years] = w_recent
        weights[(days > 365 * recent_years) & (days <= 365 * mid_years)] = w_mid
        return weights

    def update_lgbm_params(self, params: Dict) -> None:
        """次の学習で使う LightGBM パラメータ（``model.lightgbm``）を上書きする。"""
        self.config.setdefault('lightgbm', {}).update(params)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
        sample_weight: Optional[np.ndarray] = None,
    ) -> lgb.Booster:
        """二値分類モデルを学習する（``X_val`` で early stopping）。"""
        logger.info(f"モデル学習開始: {model_name}")
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weight)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = dict(self.config.get('lightgbm', {}))
        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)

        # 正例が少ない（1着は約7%）ためクラス不均衡を補正する
        pos_rate = float(y_train.mean())
        if 'is_unbalance' not in params and 0 < pos_rate < 0.15:
            params['scale_pos_weight'] = (1.0 - pos_rate) / pos_rate
            logger.info(
                f"[{model_name}] 正例率={pos_rate:.3f} "
                f"→ scale_pos_weight={params['scale_pos_weight']:.1f}"
            )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=100),
            ],
        )
        self.models[model_name] = model
        logger.info(f"モデル学習完了: {model_name} 最良イテレーション={model.best_iteration}")
        return model

    def calibrate_model(
        self,
        model_name: str,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
        method: str = 'isotonic',
    ) -> None:
        """学習済みモデルの出力確率を校正する（``'isotonic'`` または ``'sigmoid'``）。

        Raises:
            ValueError: モデルが未学習の場合
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"モデル '{model_name}' が見つかりません")

        raw_proba = model.predict(X_cal, num_iteration=model.best_iteration)
        if method == 'sigmoid':
            calibrator = LogisticRegression()
            calibrator.fit(raw_proba.reshape(-1, 1), y_cal.values)
        else:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(raw_proba, y_cal.values)
        self.calibrators[model_name] = (calibrator, method)

        # 過大評価が残ると EV が歪むため、校正後の平均予測を実測と突き合わせる
        calibrated = self._apply_calibrator(calibrator, method, raw_proba)
        mean_pred = float(np.mean(calibrated))
        actual = float(np.mean(y_cal.to_numpy(dtype=float)))
        n_unique = int(np.unique(np.round(calibrated, 6)).size)
        ratio = mean_pred / actual if actual > 0 else float('nan')
        logger.info(
            f"[{model_name}] キャリブレーション完了 method={method} 標本={len(y_cal)} "
            f"平均予測={mean_pred:.4f} 実測={actual:.4f} 比={ratio:.3f} "
            f"ユニーク確率値={n_unique}"
        )
        if not 0.9 <= ratio <= 1.1:
            logger.warning(
                f"[{model_name}] キャリブレーション後も平均予測が実測の {ratio:.2f} 倍です。"
                "EV が系統的に歪みます。"
            )
        if n_unique < 50:
            logger.warning(
                f"[{model_name}] キャリブレーション後の確率が {n_unique} 種類しかありません"
                "（量子化）。holdout 期間を長くするか method='sigmoid' を検討してください。"
            )

    @classmethod
    def _apply_calibrator(
        cls,
        calibrator: 'LogisticRegression | IsotonicRegression',
        method: str,
        proba: np.ndarray,
    ) -> np.ndarray:
        """校正器を確率に適用する。isotonic の 0.0 は下限値に切り上げる。"""
        if method == 'sigmoid':
            return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
        return np.maximum(calibrator.predict(proba), cls.CALIBRATED_PROBA_FLOOR)

    def predict(self, X: pd.DataFrame, model_name: str) -> np.ndarray:
        """予測確率を返す（校正器があれば校正後の値）。

        Raises:
            ValueError: モデルが存在しない場合
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")
        model = self.models[model_name]
        # early stopping を使わない学習では best_iteration が 0 / -1 になるため全木を使う
        best_iter = model.best_iteration
        num_iter = best_iter if best_iter and best_iter > 0 else None
        predictions = model.predict(X, num_iteration=num_iter)
        if model_name in self.calibrators:
            calibrator, method = self.calibrators[model_name]
            predictions = self._apply_calibrator(calibrator, method, predictions)
        return predictions

    def save_model(self, model_name: str, filename: str) -> None:
        """モデルと校正器（``<stem>_calibrator.pkl``）を ``model_dir`` に保存する。"""
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")
        output_path = self.model_dir / filename
        self.models[model_name].save_model(str(output_path))
        logger.info(f"モデル保存完了: {output_path}")

        if model_name in self.calibrators:
            calibrator_path = output_path.with_name(f"{output_path.stem}_calibrator.pkl")
            with open(calibrator_path, 'wb') as f:
                pickle.dump(self.calibrators[model_name], f)
            logger.info(f"キャリブレータ保存完了: {calibrator_path}")

    def load_model(self, filename: str, model_name: str) -> None:
        """``model_dir`` からモデルと対の校正器を読み込む。

        Raises:
            FileNotFoundError: モデルファイルが無い場合
        """
        model_path = self.model_dir / filename
        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        _check_model_line_endings(model_path)
        self.models[model_name] = lgb.Booster(model_file=str(model_path))
        logger.info(f"モデル読み込み完了: {model_path}")

        calibrator_path = model_path.with_name(f"{model_path.stem}_calibrator.pkl")
        if calibrator_path.exists():
            with open(calibrator_path, 'rb') as f:
                self.calibrators[model_name] = pickle.load(f)
            logger.info(f"キャリブレータ読み込み完了: {calibrator_path}")

    def get_feature_importance(
        self, model_name: str, importance_type: str = 'gain', top_n: Optional[int] = 20
    ) -> pd.DataFrame:
        """特徴量重要度を降順で返す。"""
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")
        model = self.models[model_name]
        importance_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type=importance_type),
        }).sort_values('importance', ascending=False)
        return importance_df.head(top_n) if top_n else importance_df
