"""
LightGBM 学習・予測・保存モジュール

モデルの学習・キャリブレーション・予測・保存・読込を担う。
CV / HPO / ブレンド最適化は optimizer.py に分離されている。
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.models.ranking_label import (
    compute_dynamic_odds_thresholds,
    make_relevance_labels,
)

logger = logging.getLogger(__name__)

# カテゴリ変数として LightGBM に渡すカラム候補（順序に意味がないもの）
_CATEGORICAL_FEATURE_CANDIDATES = [
    'track_type_encoded',
    'race_class_encoded',
    'best_class_encoded',
    'racecourse_encoded',
    'weather_encoded',
    'track_condition_encoded',
    'age_category_encoded',
    'sex_encoded',
]


def _check_model_line_endings(model_path: Path) -> None:
    """LightGBM モデルファイルの改行コードが LF であることを検証する。

    Windows で ``core.autocrlf`` が有効（未設定時の既定）だと、checkout 時に
    モデルファイルが LF -> CRLF へ書き換えられる。LightGBM の C++ パーサは
    ``"tree\\r"`` を解釈できず ``Model format error, expect a tree here`` を出して
    **プロセスごと異常終了する**（Python の例外にならず exit 127 になるため、
    呼び出し側は原因を掴めないまま学習が中断する）。

    ここで先に検出して、通常の Python 例外として停止させる。

    Args:
        model_path: モデルファイルのパス

    Raises:
        ValueError: ファイルが CRLF を含む場合（修復方法を併記する）
    """
    with open(model_path, 'rb') as f:
        head = f.read(4096)
    if b'\r\n' not in head:
        return
    raise ValueError(
        f"モデルファイルの改行コードが CRLF です: {model_path}\n"
        "LightGBM は CRLF のモデルを読めず、プロセスごと異常終了します"
        "（exit 127 / Model format error, expect a tree here）。\n"
        "原因: Windows の git core.autocrlf による改行変換。\n"
        "対処: .gitattributes に `data/model/**/*.txt -text` を追加した上で、\n"
        "      python -c \"import glob;[open(p,'wb').write("
        "open(p,'rb').read().replace(b'\\r\\n',b'\\n')) for p in glob.glob('data/model/*.txt')]\""
    )


def _get_categorical_cols(X: pd.DataFrame) -> list[str]:
    """X に実際に存在するカテゴリ特徴量カラムを返す。"""
    return [c for c in _CATEGORICAL_FEATURE_CANDIDATES if c in X.columns]


class LightGBMTrainer:
    """
    LightGBMモデルの学習・予測を管理するクラス。

    責務:
        - バイナリ分類モデル（単勝・複勝）の学習・キャリブレーション・予測
        - ランキングモデル（LambdaRank）の学習・予測
        - モデルの保存・読込・特徴量重要度取得
    """

    def __init__(self, config: dict, features_def: dict | None = None):
        """
        初期化

        Args:
            config: 設定辞書
            features_def: 特徴量定義辞書（features.json の内容）。省略時はクラス変数をフォールバックとして使用。
        """
        self.config = config
        self.model_config = config.get('lightgbm', {})
        self.model_dir = Path(config.get('model_dir', 'data/model'))
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.models: Dict[str, lgb.Booster] = {}
        self.calibrators: Dict[str, tuple] = {}
        # モデル名 → 回収率最良イテレーション（profit callback が記録）。
        # use_profit_iter=True の predict で参照し、save/load_model で永続化する。
        self.profit_iterations: Dict[str, int] = {}
        # モデル名 → custom fobj（Phase 2b）で学習したか。
        # fobj 使用時 LightGBM は objective を 'none' として保存し、
        # Booster.predict() は常にロジット（raw score）を返すため、
        # predict() 側で明示的に sigmoid 変換する必要がある。
        self._fobj_models: set[str] = set()
        # init_score（残差学習）で学習したモデル名。predict() が返すのは
        # 初期スコアを含まない残差のため、呼び出し側での復元が必要になる。
        self._init_score_models: set[str] = set()
        self._features_def = features_def or {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        test_size_months: int = 6
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        データを学習用・テスト用に分割

        Args:
            df: データフレーム
            feature_cols: 特徴量カラムのリスト
            target_col: ターゲットカラム名
            test_size_months: テストデータの期間（月）

        Returns:
            Tuple: (X_train, X_test, y_train, y_test, train_dates)
                train_dates は時間減衰ウェイト計算用の学習データ日付列。
        """
        logger.info("データ分割開始")

        df = df.sort_values('date').reset_index(drop=True)

        split_date = self.compute_split_date(df, test_size_months)

        train_df = df[df['date'] < split_date].copy()
        test_df = df[df['date'] >= split_date].copy()

        if len(train_df) == 0:
            total_months = (df['date'].max() - df['date'].min()).days / 30
            raise ValueError(
                f"学習データが0件です。データ期間（{total_months:.1f}ヶ月）が短すぎます。"
                f"最低でも3ヶ月以上のデータが必要です。"
            )

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        logger.info(f"分割日: {split_date.date()}")
        logger.info(f"学習データ: {len(X_train)} 件 ({len(train_df) / len(df) * 100:.1f}%)")
        logger.info(f"テストデータ: {len(X_test)} 件 ({len(test_df) / len(df) * 100:.1f}%)")
        logger.info(f"正例率（学習）: {y_train.mean():.4f}")
        logger.info(f"正例率（テスト）: {y_test.mean():.4f}")

        train_dates = train_df['date'].reset_index(drop=True)
        return X_train, X_test, y_train, y_test, train_dates

    @staticmethod
    def compute_split_date(df: pd.DataFrame, test_size_months: int = 6) -> pd.Timestamp:
        """学習/テスト分割日を計算する。

        テスト期間が全体の30%を超える場合は自動で縮小する。
        prepare_data と評価用テスト抽出の分割日を厳密に一致させ、
        学習データの評価混入（時系列リーク）を防ぐために共有する。

        Args:
            df: 'date' カラムを持つ DataFrame
            test_size_months: テストデータの期間（月）

        Returns:
            pd.Timestamp: この日付以降がテストデータ
        """
        total_months = (df['date'].max() - df['date'].min()).days / 30
        max_test_months = total_months * 0.3
        if test_size_months > max_test_months:
            original = test_size_months
            test_size_months = max(1, int(max_test_months))
            logger.warning(
                f"テストデータサイズが大きすぎます（指定: {original}ヶ月、"
                f"データ期間: {total_months:.1f}ヶ月）"
            )
            logger.warning(f"テストデータサイズを {test_size_months}ヶ月 に自動調整しました")
        return df['date'].max() - pd.DateOffset(months=test_size_months)

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
            dates: 各サンプルのレース日付 (pd.Series[datetime])
            recent_years: 直近として扱う期間（年）
            mid_years: 中期として扱う上限（年）
            w_recent: 直近期間のウェイト倍率
            w_mid: 中期のウェイト倍率

        Returns:
            np.ndarray: 各サンプルのサンプルウェイト
        """
        reference_date = dates.max()
        days = (reference_date - dates).dt.days
        weights = np.ones(len(dates), dtype=np.float32)
        weights[days <= 365 * recent_years] = w_recent
        weights[(days > 365 * recent_years) & (days <= 365 * mid_years)] = w_mid
        return weights

    @staticmethod
    def compute_odds_weights(
        odds: np.ndarray,
        gamma: float = 0.5,
        w_min: float = 0.5,
        w_max: float = 5.0,
    ) -> np.ndarray:
        """オッズ由来のサンプルウェイトを計算する（高オッズ的中を重視）。

        回収率は「的中時にオッズ倍の払戻を得る」ため、高オッズ馬の的中は
        回収率へのインパクトが大きい。これを学習に織り込むため、
        純利益倍率 ``odds - 1`` の ``gamma`` 乗をサンプルウェイトにする。

            w = clip((odds - 1) ** gamma, w_min, w_max)

        ``gamma=0`` で全て 1（従来の等重み binary と一致）。

        IMPORTANT (データリーク防止):
            ここに渡すオッズは**予測時点で既知の締切前オッズ**でなければならない。
            確定払戻（結果）を重みに使うと事実上のラベルリークになる。

        Args:
            odds: 締切前オッズの配列
            gamma: 重み付けの強さ（0 で等重み）
            w_min: ウェイトの下限（極端に低いオッズの潰れを防ぐ）
            w_max: ウェイトの上限（極端な高オッズの暴れを抑える）

        Returns:
            np.ndarray: 各サンプルのオッズウェイト（float32）

        Raises:
            ValueError: gamma が負、または w_min > w_max の場合
        """
        if gamma < 0:
            raise ValueError(f"gamma は 0 以上である必要があります: {gamma}")
        if w_min > w_max:
            raise ValueError(f"w_min <= w_max である必要があります: {w_min} > {w_max}")

        odds_arr = np.asarray(odds, dtype=float)
        if gamma == 0:
            return np.ones(len(odds_arr), dtype=np.float32)

        # 締切前オッズが欠損・不正な行は等重み（1.0）に倒す。
        # ここで停止しないのは、学習サンプル全体を落とすと期間が偏るため。
        # 欠損件数は呼び出し側（model_creator）でログ出力する。
        net = np.where(np.isfinite(odds_arr) & (odds_arr > 1.0), odds_arr - 1.0, np.nan)
        weights = np.power(net, gamma)
        weights = np.clip(weights, w_min, w_max)
        return np.where(np.isnan(weights), 1.0, weights).astype(np.float32)

    @staticmethod
    def compute_auto_pos_weight(
        model_name: str, y_train: pd.Series, params: dict
    ) -> float:
        """正例が少ないクラス不均衡データに対する pos_weight（scale_pos_weight 相当）を計算する。

        LightGBM 組み込みの ``scale_pos_weight`` は fobj 使用時に無視されるため、
        ``make_odds_weighted_fobj`` に明示的に渡す ``pos_weight`` として共通利用する
        （``train()`` の最終学習と Optuna HPO の custom fobj trial の両方で同じ値にするため）。

        win モデルかつ ``is_unbalance`` が未指定、正例率が 15% 未満の場合のみ
        ``(1 - pos_rate) / pos_rate`` を返す。それ以外は 1.0（無効）。

        Args:
            model_name: 'win' / 'place' / 'umaren'
            y_train: 学習ラベル
            params: 学習パラメータ（``is_unbalance`` の有無を見るだけ、変更しない）

        Returns:
            float: pos_weight（1.0 で無効）
        """
        pos_rate = float(y_train.mean())
        if model_name == 'win' and 'is_unbalance' not in params and 0 < pos_rate < 0.15:
            return (1.0 - pos_rate) / pos_rate
        return 1.0

    @staticmethod
    def make_odds_weighted_fobj(
        sample_weight: np.ndarray,
        y_true: np.ndarray,
        beta: float = 0.0,
        pos_weight: float = 1.0,
    ):  # -> LightGBM objective 関数（callable）
        """オッズ重み付き・非対称化ロジスティック損失の custom fobj を生成する。

        Phase 2a の ``sample_weight``（``lgb.Dataset(weight=...)``）と同じ重み
        ``w_i`` を用いるが、真の ``fobj`` にすることで **非対称化**（focal-loss 的な
        取りこぼし=偽陰性の重視）を追加で織り込める。``sample_weight`` 版だけでは
        表現できない非標準損失であり、これが Phase 2b の存在意義。

        標準の重み付きロジスティック損失:
            p_i = sigmoid(pred_i)
            grad_i = w_i * (p_i - y_i)
            hess_i = w_i * p_i * (1 - p_i)

        非対称化（``beta > 0``）では、的中馬（``y_i == 1``）を取りこぼした
        （``p_i`` が低い＝偽陰性）場合の勾配を ``(1 - p_i) ** beta`` 倍に増幅する。
        高オッズ的中の取りこぼしは回収率への機会損失が大きいため、これを重く罰する。
        ``beta == 0`` のとき、Phase 2a の ``sample_weight`` 版と数学的に等価になる。

        Args:
            sample_weight: ``compute_odds_weights`` 等で計算済みのオッズ重み ``w_i``
                （時間減衰との合成済みでよい）。**締切前オッズのみ**から計算すること
                （確定払戻を使うとラベルリーク）。
            y_true: 正解ラベル（0/1）。学習データと同じ順序・長さ。
            beta: 非対称化の強さ（0 で無効・Phase 2a と等価）。
            pos_weight: 正例（y=1）に掛けるクラス不均衡調整係数。
                LightGBM 標準の ``scale_pos_weight`` は fobj 使用時に無視されるため、
                同等の効果をここで明示的に持たせる（既定 1.0＝無効）。

        Returns:
            Callable[[np.ndarray, lgb.Dataset], Tuple[np.ndarray, np.ndarray]]:
                LightGBM の ``objective`` に渡す ``fobj`` 関数（grad, hess を返す）

        Raises:
            ValueError: beta または pos_weight が負、
                もしくは sample_weight と y_true の長さが不一致の場合
        """
        if beta < 0:
            raise ValueError(f"beta は 0 以上である必要があります: {beta}")
        if pos_weight < 0:
            raise ValueError(f"pos_weight は 0 以上である必要があります: {pos_weight}")
        w = np.asarray(sample_weight, dtype=np.float64)
        y = np.asarray(y_true, dtype=np.float64)
        if len(w) != len(y):
            raise ValueError(
                f"sample_weight と y_true の長さが一致しません: {len(w)} != {len(y)}"
            )
        # scale_pos_weight 相当を正例側にのみ適用
        class_weight = np.where(y == 1, pos_weight, 1.0)

        def fobj(preds: np.ndarray, train_data: lgb.Dataset):
            # LightGBM の fobj には常にロジット（raw score）が渡される
            p = 1.0 / (1.0 + np.exp(-preds))
            p = np.clip(p, 1e-15, 1.0 - 1e-15)

            if beta > 0:
                # 的中（y=1）を取りこぼした（p が低い）ほど勾配を増幅する。
                # 非的中（y=0）側は等倍のまま（偽陽性の罰は Phase 2a と同じ）。
                fn_penalty = np.where(y == 1, np.power(1.0 - p, beta), 1.0)
            else:
                fn_penalty = 1.0

            weight = w * fn_penalty * class_weight
            grad = weight * (p - y)
            hess = weight * p * (1.0 - p)
            return grad, hess

        return fobj

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
        init_score: Optional[np.ndarray] = None,
        init_score_val: Optional[np.ndarray] = None,
    ) -> lgb.Booster:
        """
        モデルを学習

        Args:
            X_train: 学習データの特徴量
            y_train: 学習データのターゲット
            X_val: 検証データの特徴量（オプション）
            y_val: 検証データのターゲット（オプション）
            model_name: モデル名
            sample_weight: 各サンプルのウェイト（オプション）。
                時間減衰ウェイトを使う場合は
                ``compute_time_decay_weights`` で生成したものを渡す。
                オッズ重み（高オッズ重視）を併用する場合は
                ``compute_odds_weights`` の戻り値を掛け合わせて渡す。
                ``fobj_beta`` を指定した場合、この重みは ``lgb.Dataset`` ではなく
                custom fobj 内で使われる（二重適用を避けるため Dataset には渡さない）。
            val_df_for_profit: 回収率コールバック用の検証メタ DataFrame（オプション）。
                締切前オッズ・確定払戻列を含む必要がある。指定すると回収率が最良の
                イテレーションを ``profit_iterations[model_name]`` に記録する。
            fobj_beta: Phase 2b 用。非対称化の強さ。None（既定）なら標準の binary
                objective + ``sample_weight``（Phase 2a 相当）を使う。0 以上を指定すると
                ``make_odds_weighted_fobj`` の custom fobj を使う
                （``sample_weight`` はこの fobj 内の重みとして使われ、``0`` なら Phase 2a と等価）。
                custom fobj 使用時は ``objective`` を上書きするため、
                early stopping 用の metric として AUC を明示的に併用する。
            init_score: 学習データの初期スコア（base margin, logit スケール）。
                市場アンカー型の残差学習（``market_edge_analysis.md`` §5 P1）で
                ``logit(p_market)`` を渡すと、葉は
                ``logit(p_true) - logit(p_market)`` のみを学習する。
                特徴量として渡す場合と異なり、**構造的に市場を再現できない**ため
                乖離だけを表現するモデルになる。
                指定時は ``predict`` の出力にこの初期スコアが含まれない点に注意
                （呼び出し側で ``sigmoid(init_score + raw_score)`` を復元すること）。
            init_score_val: 検証データの初期スコア。``X_val`` 指定時は
                ``init_score`` と対で渡す必要がある（early stopping の評価が
                学習と同じスケールで行われないと不整合になるため）。

        Returns:
            lgb.Booster: 学習済みモデル

        Raises:
            ValueError: ``init_score`` の長さが学習データと一致しない場合、
                または ``X_val`` があるのに ``init_score_val`` が欠けている場合
        """
        logger.info(f"モデル学習開始: {model_name}")

        cat_cols = _get_categorical_cols(X_train)
        use_fobj = fobj_beta is not None
        # fobj 使用時は重みを fobj 内部で適用するため Dataset には渡さない（二重適用防止）
        dataset_weight = None if use_fobj else sample_weight

        if init_score is not None:
            if len(init_score) != len(y_train):
                raise ValueError(
                    f"[{model_name}] init_score の長さが学習データと一致しません: "
                    f"{len(init_score)} != {len(y_train)}"
                )
            if X_val is not None and init_score_val is None:
                raise ValueError(
                    f"[{model_name}] X_val があるのに init_score_val が指定されていません。"
                    "検証データにも同じ初期スコアを与えないと early stopping が"
                    "学習と異なるスケールで評価され不整合になります。"
                )
            logger.info(
                f"[{model_name}] init_score（残差学習）を使用: "
                f"mean={float(np.mean(init_score)):.4f}"
            )

        train_data = lgb.Dataset(
            X_train, label=y_train, weight=dataset_weight,
            categorical_feature=cat_cols, init_score=init_score,
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            if init_score_val is not None and len(init_score_val) != len(y_val):
                raise ValueError(
                    f"[{model_name}] init_score_val の長さが検証データと一致しません: "
                    f"{len(init_score_val)} != {len(y_val)}"
                )
            val_data = lgb.Dataset(
                X_val, label=y_val, reference=train_data, init_score=init_score_val
            )
            valid_sets.append(val_data)
            valid_names.append('valid')

        params = self.model_config.copy()
        # モデルごとのパラメータ上書き（place / umaren は専用セクションを持つ）
        override_key = {'place': 'lightgbm_place', 'umaren': 'lightgbm_umaren'}.get(model_name)
        if override_key:
            params.update(self.config.get(override_key, {}))

        # scale_pos_weight は LightGBM 組み込み binary objective 専用パラメータ。
        # fobj 使用時は自動適用されないため、後段で pos_weight として fobj に明示的に渡す。
        auto_pos_weight = self.compute_auto_pos_weight(model_name, y_train, params)
        if auto_pos_weight != 1.0 and not use_fobj:
            params['scale_pos_weight'] = auto_pos_weight
            logger.info(
                f"[{model_name}] 正例率={float(y_train.mean()):.3f} "
                f"→ scale_pos_weight={auto_pos_weight:.1f}"
            )

        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)

        if use_fobj:
            # custom fobj は params['objective'] を上書きする。LightGBM は fobj 使用時
            # metric を自動推定しないため、early stopping / ログ用に明示的に指定する。
            beta = float(fobj_beta)  # type: ignore[arg-type]
            weight_for_fobj = (
                np.ones(len(y_train), dtype=np.float64)
                if sample_weight is None
                else np.asarray(sample_weight, dtype=np.float64)
            )
            params.pop('scale_pos_weight', None)
            params.pop('is_unbalance', None)
            params['objective'] = self.make_odds_weighted_fobj(
                weight_for_fobj, y_train.to_numpy(), beta=beta, pos_weight=auto_pos_weight,
            )
            params['metric'] = 'auc'
            logger.info(
                f"[{model_name}] Phase 2b custom fobj を使用: beta={beta} "
                f"pos_weight={auto_pos_weight:.2f}"
            )

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=100)
        ]
        # 回収率が最良のイテレーションを記録する（AUC 最良とは別に選択できるようにする）
        if val_df_for_profit is not None and X_val is not None:
            callbacks.append(
                self._make_profit_callback(
                    val_df_for_profit,
                    list(X_val.columns),
                    model_name=model_name,
                    model_kind='binary',
                    bet_type=model_name,
                )
            )

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.models[model_name] = model
        if use_fobj:
            self._fobj_models.add(model_name)
        else:
            self._fobj_models.discard(model_name)
        if init_score is not None:
            self._init_score_models.add(model_name)
        else:
            self._init_score_models.discard(model_name)

        logger.info(f"モデル学習完了: {model_name}")
        logger.info(f"最良イテレーション: {model.best_iteration}")
        logger.info(f"最良スコア: {model.best_score}")

        return model

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

        Platt scaling（sigmoid）または Isotonic Regression を使って
        LightGBM の生確率を補正し、EV 計算の精度を向上させる。

        IMPORTANT:
            キャリブレーション標本が少ないと isotonic が過学習し、出力確率が
            少数の離散値に量子化されて EV が系統的に過大評価される。
            ``min_samples_per_bin`` で1プラトーあたりの最小標本数を担保すること。

        IMPORTANT:
            正例率が低い券種（馬連は ~6.7%）では、件数だけでビンを切ると
            下位ビンの正例が0件になり ``bin_y = 0`` が連続する。isotonic は
            単調制約下でこれを1本の「出力が厳密に 0.0」のプラトーとして
            再現するため、EV = proba × odds がゼロになりモデルの順序情報が
            失われる。``min_pos_per_bin`` で各ビンに正例を最低数確保すること。
            詳細: output/20260811_06_umaren_zero_proba_investigation.md

        Args:
            model_name: キャリブレーション対象のモデル名
            X_cal: キャリブレーション用特徴量（バリデーションセット）
            y_cal: キャリブレーション用ラベル（バリデーションセット）
            method: 'isotonic'（デフォルト）または 'sigmoid'（Platt scaling）
            min_samples_per_bin: isotonic の1プラトーあたり最小標本数（0 で無効）
            min_pos_per_bin: isotonic の1ビンあたり最小正例数（0 で無効）。
                正例0件ビンの連続による 0.0 プラトーを防ぐ。

        Raises:
            ValueError: モデルが未学習の場合
        """
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"モデル '{model_name}' が見つかりません")

        raw_proba = model.predict(X_cal, num_iteration=model.best_iteration)
        if model_name in self._fobj_models:
            # custom fobj（Phase 2b）モデルは model.predict() がロジットを返す。
            # calibrator は predict() 側（sigmoid 変換後の確率）を入力として使われるため、
            # fit 時も同じスケール（確率）に揃えないと入力ドメインが食い違い、
            # キャリブレーションが機能しない（isotonic の学習域が [0,1] に対し
            # 実際の入力がロジットの広い範囲になり、範囲外は端点にクリップされ続ける）。
            raw_proba = 1.0 / (1.0 + np.exp(-raw_proba))

        if method == 'sigmoid':
            calibrator: LogisticRegression | IsotonicRegression = LogisticRegression()
            calibrator.fit(raw_proba.reshape(-1, 1), y_cal.values)
        else:
            calibrator = IsotonicRegression(out_of_bounds='clip')
            if min_samples_per_bin > 0:
                # 標本を等頻度ビンに集約してから fit する。isotonic は入力点ごとに
                # プラトーを作るため、生の (proba, 0/1) をそのまま渡すと 1/3 や 6/11 の
                # ような小標本の的中率がそのまま出力値になる。各ビンに
                # min_samples_per_bin 件以上を確保し、ビン内平均を教師にすることで
                # プラトー数を抑え、各出力値の推定精度を担保する。
                n_bins = max(1, len(raw_proba) // min_samples_per_bin)
                bin_x, bin_y, bin_w = self._aggregate_for_isotonic(
                    raw_proba, y_cal.to_numpy(dtype=float), n_bins,
                    min_pos_per_bin=min_pos_per_bin,
                )
                calibrator.fit(bin_x, bin_y, sample_weight=bin_w)
            else:
                calibrator.fit(raw_proba, y_cal.values)

        self.calibrators[model_name] = (calibrator, method)

        # キャリブレーション品質を検証する。過大評価が残っていれば EV 計算が歪むため、
        # 学習ログの時点で気づけるように実測値と突き合わせて出力する。
        calibrated = self._apply_calibrator(calibrator, method, raw_proba)
        mean_pred = float(np.mean(calibrated))
        actual = float(np.mean(y_cal.to_numpy(dtype=float)))
        n_unique = int(np.unique(np.round(calibrated, 6)).size)
        ratio = mean_pred / actual if actual > 0 else float('nan')
        logger.info(
            f"キャリブレーション完了: model='{model_name}' method={method} "
            f"calibration_samples={len(y_cal)}"
        )
        logger.info(
            f"[{model_name}] キャリブレーション品質: 平均予測={mean_pred:.4f} "
            f"実測={actual:.4f} 比={ratio:.3f} ユニーク確率値={n_unique}"
        )
        if not 0.9 <= ratio <= 1.1:
            logger.warning(
                f"[{model_name}] キャリブレーション後も平均予測が実測の {ratio:.2f} 倍です。"
                "EV が系統的に歪むため calibration.months / method を見直してください。"
            )
        if n_unique < 50:
            logger.warning(
                f"[{model_name}] キャリブレーション後の確率が {n_unique} 種類しかありません"
                "（量子化）。calibration.months を増やすか method='sigmoid' を検討してください。"
            )
        self._warn_if_zero_plateau(model_name, calibrator, method)

    # 校正後に確率が厳密に 0.0 となる入力レンジがこの割合を超えたら警告する。
    ZERO_PLATEAU_WARN_RATIO = 0.30

    # isotonic 校正後の確率の下限。EV=proba×odds が恒久的に 0 になるのを防ぐ。
    # 「0.0 は賭けない」を「賭ける」に反転させてはならないため、馬連の実測
    # 最大オッズ 12,000 倍でも EV が最小閾値 1.2 に届かない水準に取る
    # （1e-5 × 12,000 = 0.12 << 1.2）。あくまで恒久的なゼロ乗算を避ける
    # 安全網であり、確率としての意味は持たせない。
    CALIBRATED_PROBA_FLOOR = 1e-5

    def _warn_if_zero_plateau(
        self,
        model_name: str,
        calibrator: 'LogisticRegression | IsotonicRegression',
        method: str,
    ) -> None:
        """校正器が広い「出力 0.0」プラトーを持つ場合に警告する。

        isotonic は下位ビンの正例が0件だと出力を厳密に 0.0 にする。
        EV = proba × odds は乗法的なので、0.0 になった候補はオッズが
        何倍でも永久に選ばれず、その入力レンジのモデル順序情報が失われる。
        馬連では入力レンジの 58% が 0.0 に潰れていた（2026-08-11 調査）。

        Args:
            model_name: モデル名（ログ用）
            calibrator: 学習済み calibrator
            method: 'isotonic' または 'sigmoid'
        """
        if method != 'isotonic':
            return
        xt = getattr(calibrator, 'X_thresholds_', None)
        yt = getattr(calibrator, 'y_thresholds_', None)
        if xt is None or yt is None or len(xt) < 2:
            return

        span = float(xt.max() - xt.min())
        if span <= 0:
            return
        zero_idx = np.nonzero(yt == 0.0)[0]
        if len(zero_idx) == 0:
            return
        # 単調なので先頭から連続する 0 の右端がプラトーの上限になる
        zero_upto = float(xt[zero_idx.max()])
        ratio = (zero_upto - float(xt.min())) / span
        if ratio > self.ZERO_PLATEAU_WARN_RATIO:
            logger.warning(
                f"[{model_name}] キャリブレーション後、生確率 {zero_upto:.4f} 以下が"
                f"すべて 0.0 に潰れます（入力レンジの {ratio * 100:.1f}%）。"
                "EV=proba×odds が 0 になりモデルの順序情報が失われます。"
                "calibration.min_pos_per_bin を設定するか method='sigmoid' を検討してください。"
            )

    @staticmethod
    def _aggregate_for_isotonic(
        raw_proba: np.ndarray,
        y: np.ndarray,
        n_bins: int,
        min_pos_per_bin: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """isotonic 用に生確率を等頻度ビンへ集約する。

        ``min_pos_per_bin > 0`` の場合、等頻度ビンを確率の低い側から走査し、
        正例数が満たない間は隣のビンへ併合する。これにより
        「正例0件のビンが連続して bin_y=0 のプラトーを作る」事態を防ぐ。
        正例率が低い券種（馬連）で isotonic の出力が厳密に 0.0 に潰れるのを
        避けるための制約である。

        Args:
            raw_proba: モデルの生確率
            y: 正解ラベル（0/1）
            n_bins: ビン数
            min_pos_per_bin: 1ビンあたり最小正例数（0 で無効）

        Returns:
            tuple: (ビン代表確率, ビン内正例率, ビン内件数)
        """
        order = np.argsort(raw_proba, kind='mergesort')
        p_sorted = raw_proba[order]
        y_sorted = y[order]
        # 端数はできるだけ均等に分配する（np.array_split と同じ分け方）
        chunks_p = [c for c in np.array_split(p_sorted, n_bins) if len(c) > 0]
        chunks_y = [c for c in np.array_split(y_sorted, n_bins) if len(c) > 0]

        if min_pos_per_bin > 0:
            chunks_p, chunks_y = LightGBMTrainer._merge_bins_by_positives(
                chunks_p, chunks_y, min_pos_per_bin
            )

        bin_x = np.array([c.mean() for c in chunks_p])
        bin_y = np.array([c.mean() for c in chunks_y])
        bin_w = np.array([float(len(c)) for c in chunks_p])
        return bin_x, bin_y, bin_w

    @staticmethod
    def _merge_bins_by_positives(
        chunks_p: list[np.ndarray],
        chunks_y: list[np.ndarray],
        min_pos_per_bin: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """正例数が ``min_pos_per_bin`` に満たないビンを隣へ併合する。

        確率の低い側から貪欲に積み上げ、正例数が閾値に達した時点でビンを確定する。
        最後に余ったビン（正例が足りないまま末尾に到達した分）は、直前の
        確定済みビンへ吸収する。1本も確定できない場合は全体を1ビンにする。

        Args:
            chunks_p: 確率昇順に並んだビンごとの生確率配列
            chunks_y: ``chunks_p`` と対応するラベル配列
            min_pos_per_bin: 1ビンあたり最小正例数

        Returns:
            tuple: 併合後の (確率配列リスト, ラベル配列リスト)
        """
        merged_p: list[list[np.ndarray]] = []
        merged_y: list[list[np.ndarray]] = []
        cur_p: list[np.ndarray] = []
        cur_y: list[np.ndarray] = []
        cur_pos = 0.0

        for cp, cy in zip(chunks_p, chunks_y):
            cur_p.append(cp)
            cur_y.append(cy)
            cur_pos += float(cy.sum())
            if cur_pos >= min_pos_per_bin:
                merged_p.append(cur_p)
                merged_y.append(cur_y)
                cur_p, cur_y, cur_pos = [], [], 0.0

        # 末尾に残った端数は直前のビンへ吸収する（単独では正例数を満たせないため）
        if cur_p:
            if merged_p:
                merged_p[-1].extend(cur_p)
                merged_y[-1].extend(cur_y)
            else:
                merged_p.append(cur_p)
                merged_y.append(cur_y)

        return (
            [np.concatenate(g) for g in merged_p],
            [np.concatenate(g) for g in merged_y],
        )

    @staticmethod
    def _apply_calibrator(
        calibrator: 'LogisticRegression | IsotonicRegression',
        method: str,
        proba: np.ndarray,
    ) -> np.ndarray:
        """calibrator を確率配列に適用する（predict() と同じ変換）。

        isotonic が出力しうる厳密な 0.0 は ``CALIBRATED_PROBA_FLOOR`` で
        下限クリップする。EV = proba × odds は乗法的なので、0.0 は
        「オッズが何倍でも絶対に賭けない」という無限に強い主張になってしまう。
        校正標本にたまたま正例が無かっただけのビンにその強さは正当化できない。

        なお下限クリップは 0.0 同士の順序差を復元しない（対症療法）。
        根治は ``min_pos_per_bin`` によるビン併合と、正例率が低い券種での
        ``method='sigmoid'`` 採用の側で行う。

        Args:
            calibrator: 学習済み calibrator
            method: 'isotonic' または 'sigmoid'
            proba: 変換対象の確率

        Returns:
            np.ndarray: キャリブレーション後の確率
        """
        if method == 'sigmoid':
            # Platt は原理的に厳密な 0 を出さないためクリップ不要
            return calibrator.predict_proba(proba.reshape(-1, 1))[:, 1]
        return np.maximum(calibrator.predict(proba), LightGBMTrainer.CALIBRATED_PROBA_FLOOR)

    def compute_calibration_stats(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_bins: int = 10,
    ) -> Dict:
        """
        モデルのキャリブレーション品質を計算する。

        ``sklearn.calibration.calibration_curve`` を使って
        平均予測確率 vs 実際の正例率を n_bins 分割で計算する。

        Args:
            model_name: 対象モデル名
            X: 特徴量データ
            y: 正解ラベル
            n_bins: ビン数

        Returns:
            Dict: fraction_of_positives / mean_predicted_value / brier_score を含む辞書
        """
        predictions = self.predict(X, model_name=model_name)
        fraction_pos, mean_pred = calibration_curve(y, predictions, n_bins=n_bins)
        from sklearn.metrics import brier_score_loss
        brier = brier_score_loss(y, predictions)

        logger.info(
            f"キャリブレーション統計: model='{model_name}' "
            f"Brier={brier:.4f}"
        )
        return {
            'fraction_of_positives': fraction_pos,
            'mean_predicted_value': mean_pred,
            'brier_score': brier,
        }

    def predict(
        self,
        X: pd.DataFrame,
        model_name: str = 'model',
        use_profit_iter: bool = False,
        init_score: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        予測を実行

        Args:
            X: 特徴量データ
            model_name: モデル名
            use_profit_iter: True のとき ranking モデルで回収率最大イテレーションを使用。
                ranking_best_profit_iteration が未設定の場合は best_iteration にフォールバックする。
            init_score: 残差学習モデル（``train(init_score=...)`` で学習）用の初期スコア。
                学習時と同じ ``logit(p_market)`` を渡す必要がある。
                LightGBM の ``predict`` は初期スコアを含まない残差を返すため、
                ``sigmoid(init_score + raw_score)`` として確率を復元する。

        Returns:
            np.ndarray: 予測確率

        Raises:
            ValueError: モデルが存在しない場合、残差学習モデルなのに
                ``init_score`` が渡されない場合（残差を確率と誤認するのを防ぐ）、
                または長さが一致しない場合
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")

        needs_init = model_name in self._init_score_models
        if needs_init and init_score is None:
            raise ValueError(
                f"モデル '{model_name}' は init_score（残差学習）で学習されています。"
                "predict にも学習時と同じ初期スコアを渡してください。"
                "省略すると残差そのものを確率として扱い、市場アンカーが失われます。"
            )
        if init_score is not None:
            if not needs_init:
                raise ValueError(
                    f"モデル '{model_name}' は init_score で学習されていませんが "
                    "predict に init_score が渡されました。"
                )
            if len(init_score) != len(X):
                raise ValueError(
                    f"init_score の長さが X と一致しません: {len(init_score)} != {len(X)}"
                )

        model = self.models[model_name]

        profit_iter = self.profit_iterations.get(model_name)
        if use_profit_iter and profit_iter is not None:
            num_iter = profit_iter
            logger.info(
                f"[predict] model='{model_name}' 回収率最良イテレーション {profit_iter} を使用"
            )
        else:
            # early stopping を使わない学習では best_iteration が 0 / -1 になる。
            # そのまま渡すと予測が壊れる（0 は「全木」だが -1 は不定）ため、
            # 有効な値でないときは None（＝全イテレーション）にフォールバックする。
            best_iter = model.best_iteration
            num_iter = best_iter if best_iter and best_iter > 0 else None

        if needs_init:
            # init_score 学習時、LightGBM の predict() は初期スコアを含まない
            # 残差だけを sigmoid した値を返す（実測確認済み）。確率を復元するには
            # raw_score=True で残差ロジットを取り、初期スコアを足してから sigmoid する。
            raw = model.predict(X, num_iteration=num_iter, raw_score=True)
            predictions = 1.0 / (
                1.0 + np.exp(-(np.asarray(init_score, dtype=float) + raw))
            )
        else:
            predictions = model.predict(X, num_iteration=num_iter)
        if model_name in self._fobj_models:
            # custom fobj（Phase 2b）で学習したモデルは objective='none' で保存され、
            # predict() は常にロジット（raw score）を返すため、ここで確率に変換する。
            predictions = 1.0 / (1.0 + np.exp(-predictions))

        if model_name in self.calibrators:
            calibrator, method = self.calibrators[model_name]
            predictions = self._apply_calibrator(calibrator, method, predictions)

        return predictions

    def save_model(self, model_name: str = 'model', filename: Optional[str] = None) -> None:
        """
        モデルを保存

        Args:
            model_name: モデル名
            filename: 保存ファイル名
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")

        if filename is None:
            filename = f"{model_name}.txt"

        output_path = self.model_dir / filename
        model = self.models[model_name]
        model.save_model(str(output_path))

        logger.info(f"モデル保存完了: {output_path}")

        # キャリブレータが存在する場合は対で永続化する。
        # 学習時と推論時で確率スケールが一致しないと EV 閾値・Kelly 計算が壊れるため、
        # Booster と同じ basename で <stem>_calibrator.pkl を保存する。
        if model_name in self.calibrators:
            calibrator_path = output_path.with_name(f"{output_path.stem}_calibrator.pkl")
            with open(calibrator_path, 'wb') as f:
                pickle.dump(self.calibrators[model_name], f)
            logger.info(f"キャリブレータ保存完了: {calibrator_path}")

        # 回収率最良イテレーション／custom fobj フラグを <stem>_meta.json に永続化する。
        # 別プロセス（predictor）で load 後に use_profit_iter や sigmoid 変換を機能させるため。
        is_fobj = model_name in self._fobj_models
        is_init_score = model_name in self._init_score_models
        if model_name in self.profit_iterations or is_fobj or is_init_score:
            meta: Dict = {}
            if model_name in self.profit_iterations:
                meta['profit_iteration'] = int(self.profit_iterations[model_name])
            if is_fobj:
                meta['is_fobj'] = True
            if is_init_score:
                # 残差学習モデルであることを永続化する。これが無いと別プロセスで
                # load_model したときに predict が init_score を要求せず、
                # 市場アンカーを欠いた残差そのものを確率として返してしまう。
                meta['is_init_score'] = True
            meta_path = output_path.with_name(f"{output_path.stem}_meta.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
            logger.info(f"モデルメタ保存完了: {meta_path}")

    def load_model(self, filename: str, model_name: str = 'model') -> None:
        """
        モデルを読み込み

        Args:
            filename: ファイル名
            model_name: モデル名
        """
        model_path = self.model_dir / filename

        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        _check_model_line_endings(model_path)

        model = lgb.Booster(model_file=str(model_path))
        self.models[model_name] = model

        logger.info(f"モデル読み込み完了: {model_path}")

        # 対で保存されたキャリブレータがあれば読み込む。
        # 存在すれば predict() が学習時と同じ補正確率を返す。
        calibrator_path = model_path.with_name(f"{model_path.stem}_calibrator.pkl")
        if calibrator_path.exists():
            with open(calibrator_path, 'rb') as f:
                self.calibrators[model_name] = pickle.load(f)
            logger.info(f"キャリブレータ読み込み完了: {calibrator_path}")

        # 回収率最良イテレーション／custom fobj フラグのメタがあれば復元する。
        # profit_iteration が無いと use_profit_iter=True でも best_iteration にフォールバックする。
        # is_fobj が無いと fobj モデルの predict() がロジットのまま返ってしまう。
        meta_path = model_path.with_name(f"{model_path.stem}_meta.json")
        self._fobj_models.discard(model_name)
        self._init_score_models.discard(model_name)
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            if 'profit_iteration' in meta:
                self.profit_iterations[model_name] = int(meta['profit_iteration'])
                logger.info(
                    f"回収率最良イテレーション読み込み完了: "
                    f"{self.profit_iterations[model_name]}（{meta_path}）"
                )
            if meta.get('is_fobj'):
                self._fobj_models.add(model_name)
                logger.info(f"custom fobj モデルとして読み込み完了（sigmoid変換を適用）: {model_name}")
            if meta.get('is_init_score'):
                self._init_score_models.add(model_name)
                logger.info(
                    f"残差学習モデルとして読み込み完了（predict に init_score が必須）: "
                    f"{model_name}"
                )

    def get_feature_importance(
        self,
        model_name: str = 'model',
        importance_type: str = 'gain',
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        特徴量重要度を取得

        Args:
            model_name: モデル名
            importance_type: 重要度のタイプ（'gain', 'split'）
            top_n: 上位N件を取得

        Returns:
            pd.DataFrame: 特徴量重要度
        """
        if model_name not in self.models:
            raise ValueError(f"モデル '{model_name}' が見つかりません")

        model = self.models[model_name]

        importance_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type=importance_type)
        })

        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def prepare_ranking_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ranking_label_config: dict,
        test_size_months: int = 6
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
        List[int], List[int], pd.DataFrame, pd.DataFrame, List[str]
    ]:
        """
        ランキング学習用データを準備する。

        LightGBM ランキング学習の制約:
        - 同一グループ（レース）のデータが連続している必要がある
        - group パラメータ = 各グループの行数リスト [16, 14, 18, ...]
        - label = 非負整数の関連度スコア（高い = より関連度が高い）

        オッズ加重ラベル（scheme="odds_weighted" 時）:
        - 1着かつ高配当 → 最高ラベル4（穴馬的中 = 最大回収価値）
        - 1着かつ普通オッズ → ラベル3
        - 1着かつ低オッズ → ラベル2（本命的中 = 低回収価値）
        - 3着以内かつ高配当 → ラベル2
        - 3着以内 → ラベル1
        - その他 → ラベル0

        Args:
            df: 特徴量DataFrame（race_id, date, finish_position カラムが必須）
            feature_cols: 学習に使う特徴量カラムリスト
            ranking_label_config: config['model']['ranking_label']
            test_size_months: テスト期間（月）

        Returns:
            (X_train, X_test, y_train, y_test,
             groups_train, groups_test, train_df, test_df, ranking_feature_cols)
            ranking_feature_cols は現状 feature_cols をそのまま返す（互換のための戻り値）。
        """
        logger.info("ランキング学習用データ準備開始")

        df = df.sort_values('date').reset_index(drop=True)

        # binary 分割（prepare_data）と同一の分割日を使い、win/place/ranking で
        # テスト期間を揃える。以前は Timedelta(days=30*月) で微妙にずれていた。
        split_date = self.compute_split_date(df, test_size_months)

        train_df = df[df['date'] < split_date].copy()
        test_df = df[df['date'] >= split_date].copy()

        train_df = train_df.sort_values(['date', 'race_id']).reset_index(drop=True)
        test_df = test_df.sort_values(['date', 'race_id']).reset_index(drop=True)

        # ラベル生成は make_relevance_labels に集約（CV と本学習で定義を一致させる）
        scheme = ranking_label_config.get('scheme', 'positional')
        dynamic_thr = None
        if scheme == 'odds_weighted_dynamic' and 'odds_win' in df.columns:
            # リーク防止: 閾値は train_df のみで計算する（test_df を含めると未来情報が漏れる）
            dynamic_thr = compute_dynamic_odds_thresholds(train_df)
            logger.info(
                f"動的オッズ閾値（train のみ）: mid_win={dynamic_thr['mid_win']:.1f}, "
                f"high_win={dynamic_thr['high_win']:.1f}, high_place={dynamic_thr['high_place']:.1f}"
            )
        y_train = pd.Series(
            make_relevance_labels(train_df, ranking_label_config, dynamic_thr),
            index=train_df.index,
        )
        y_test = pd.Series(
            make_relevance_labels(test_df, ranking_label_config, dynamic_thr),
            index=test_df.index,
        )

        groups_train = train_df.groupby('race_id', sort=False).size().tolist()
        groups_test = test_df.groupby('race_id', sort=False).size().tolist()

        X_train = train_df[feature_cols]
        X_test = test_df[feature_cols]

        logger.info(f"ランキング学習データ: {len(X_train)} 件 / {len(groups_train)} レース")
        logger.info(f"ランキングテストデータ: {len(X_test)} 件 / {len(groups_test)} レース")
        logger.info(f"関連度スコア分布（学習）: {pd.Series(y_train).value_counts().to_dict()}")

        return (
            X_train, X_test, y_train, y_test,
            groups_train, groups_test, train_df, test_df, feature_cols
        )

    def _make_profit_callback(
        self,
        val_df: pd.DataFrame,
        val_feature_cols: List[str],
        model_name: str = 'ranking',
        period: int = 100,
        model_kind: str = 'ranking',
        bet_type: str = 'win',
    ):  # -> LightGBM コールバック関数（ただの callable）
        """
        検証セットの回収率を定期的にログ出力するコールバックを生成する。

        AUC / NDCG ではなく回収率が最大となるイテレーションを特定するためのコールバック。
        早期終了は変更せず、回収率最良イテレーションを
        ``self.profit_iterations[model_name]`` に記録する。
        ``predict(use_profit_iter=True)`` で、その iteration を採用できる。

        賭け判断には締切前オッズ（``odds_pre_*``）、的中時の払戻には確定払戻金
        （``payout_*``）を使い、賭け判断と払戻でオッズ源を分離してリークを避ける。

        Args:
            val_df: 検証用DataFrame（締切前オッズ・確定払戻・finish_position が必要）
            val_feature_cols: 検証特徴量カラムリスト
            model_name: 記録先のモデル名（profit_iterations のキー）
            period: ログ出力間隔（イテレーション数）
            model_kind: 'ranking'（スコア→softmax確率）または 'binary'（出力がそのまま確率）
            bet_type: 賭けタイプ（'win' / 'place' / 'umaren'）

        Returns:
            LightGBM コールバック関数
        """
        from src.models.odds_series import BET_ODDS_COLS, PAYOUT_COLS

        best_state: Dict = {'recovery_rate': 0.0, 'iteration': 0, 'disabled': False}

        # クロージャ外で1回だけ準備（毎回の再import・再計算を回避）
        from src.predict.blend_ensemble import ranking_to_proba
        eval_cfg = self.config.get('evaluation', {})
        temperature = eval_cfg.get('ranking_temperature', 1.5)
        min_ev = eval_cfg.get(f'min_ev_{bet_type}', eval_cfg.get('min_ev_win', 1.05))

        bet_odds_col = BET_ODDS_COLS[bet_type]
        payout_col = PAYOUT_COLS[bet_type]
        hit_col = 'target_umaren' if bet_type == 'umaren' else 'finish_position'
        required = (bet_odds_col, payout_col, hit_col)
        has_cols = all(c in val_df.columns for c in required)
        if not has_cols:
            missing = [c for c in required if c not in val_df.columns]
            logger.warning(
                f"[profit_cb] 必要な列が不足のため回収率観測を無効化します: {missing}"
            )
        else:
            # 締切前オッズが欠損する行は評価対象外（確定オッズで代用しない）
            work = val_df.reset_index(drop=True)
            valid = work[bet_odds_col].notna() & (work[bet_odds_col] > 0)
            n_dropped = int((~valid).sum())
            if n_dropped > 0:
                logger.warning(
                    f"[profit_cb] 締切前オッズ欠損のため {n_dropped}/{len(work)} 行を"
                    "回収率観測の対象外にしました"
                )
            work = work[valid].reset_index(drop=True)
            has_cols = len(work) > 0
            if not has_cols:
                logger.warning("[profit_cb] 有効行が0件のため回収率観測を無効化します")
            else:
                val_X = work[val_feature_cols]
                odds_arr = work[bet_odds_col].to_numpy(dtype=float)
                # 払戻金は100円あたりなので倍率へ換算
                payout_arr = work[payout_col].fillna(0.0).to_numpy(dtype=float) / 100.0
                if bet_type == 'umaren':
                    hit_arr = work['target_umaren'].to_numpy() == 1
                elif bet_type == 'place':
                    # finish_position には出走取消・除外・失格を表す負値（-1/-2/-3）が
                    # 入るため、`<= 3` だけだとこれらを的中と誤判定する。
                    _pos = work['finish_position'].to_numpy()
                    hit_arr = (_pos >= 1) & (_pos <= 3)
                else:
                    hit_arr = work['finish_position'].to_numpy() == 1
                # レースごとの行インデックスを事前にグループ化
                race_groups = [
                    idx.to_numpy()
                    for _, idx in work.groupby('race_id').groups.items()
                ]

        def callback(env: lgb.callback.CallbackEnv) -> None:  # type: ignore[attr-defined]
            if env.iteration % period != 0 or not has_cols or best_state['disabled']:
                return

            try:
                scores = env.model.predict(val_X)
                total_bet = 0.0
                total_return = 0.0
                # iterrows / DataFrame.copy を廃止し、事前グループ化した配列で集計
                for gidx in race_groups:
                    if model_kind == 'ranking':
                        proba = ranking_to_proba(scores[gidx], temperature)
                    else:
                        # binary はモデル出力がそのまま確率
                        proba = scores[gidx]
                    hit_mask = proba * odds_arr[gidx] >= min_ev
                    n = int(hit_mask.sum())
                    if n == 0:
                        continue
                    total_bet += 100.0 * n
                    won = hit_mask & hit_arr[gidx]
                    total_return += 100.0 * payout_arr[gidx][won].sum()

                rr = (total_return / total_bet * 100) if total_bet > 0 else 0.0

                if rr > best_state['recovery_rate']:
                    best_state['recovery_rate'] = rr
                    best_state['iteration'] = env.iteration
                    # モデル名ごとの辞書に記録（save/load で永続化される）。
                    # 後方互換のため単一属性も更新する。
                    self.profit_iterations[model_name] = env.iteration
                    self.ranking_best_profit_iteration = env.iteration
                    self.ranking_best_profit_recovery_rate = rr

                logger.info(
                    f"[profit_cb] iter={env.iteration} "
                    f"recovery_rate={rr:.1f}% "
                    f"(best={best_state['recovery_rate']:.1f}%@{best_state['iteration']})"
                )
            except Exception as e:
                # 実バグを握り潰さないよう warning で記録し、以後はスキップする
                logger.warning(f"[profit_cb] エラーのため以後の回収率観測を無効化します: {e}")
                best_state['disabled'] = True

        callback.order = 25
        return callback

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

        バイナリ分類の train() との主な差異:
        - config['lightgbm_ranking'] を使用（objective=lambdarank）
        - lgb.Dataset に group= パラメータが必要
        - 予測出力はスコア（確率ではない）: 高いほど上位ランク予測

        val_df_for_profit を渡すと、学習中に100イテレーションごとに
        検証セットの回収率をログ出力する利益コールバックが有効になる。

        Args:
            X_train: 学習特徴量
            y_train: 関連度スコア（非負整数）
            groups_train: 各レースの馬数リスト（data の並び順と一致）
            X_val: 検証特徴量（オプション）
            y_val: 検証ラベル（オプション）
            groups_val: 検証グループリスト（オプション）
            model_name: モデル識別子
            val_df_for_profit: 利益コールバック用の検証DataFrame（オプション）
            val_feature_cols_for_profit: 利益コールバック用の特徴量カラム（オプション）

        Returns:
            lgb.Booster: 学習済みランキングモデル
        """
        logger.info(f"ランキングモデル学習開始: {model_name}")

        self.ranking_best_profit_iteration: Optional[int] = None
        self.ranking_best_profit_recovery_rate: float = 0.0
        # 再学習時に古い profit iteration を残さない
        self.profit_iterations.pop(model_name, None)

        ranking_config = self.config.get('lightgbm_ranking', {})
        params = ranking_config.copy()

        n_estimators = params.pop('n_estimators', 1000)
        early_stopping_rounds = params.pop('early_stopping_rounds', 50)

        cat_cols = _get_categorical_cols(X_train)
        train_data = lgb.Dataset(
            X_train, label=y_train, group=groups_train, categorical_feature=cat_cols
        )

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None and groups_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # early_stopping_rounds <= 0 なら early stopping を使わず n_estimators まで学習する。
        # lambdarank の valid NDCG は、本学習の valid 窓が特定季節（冬季）に固定されると
        # 1イテレーション目が最良になり num_trees=1 の使い物にならないモデルが保存される
        # （2026-08-22 に発生。実測 top1的中率 27.7% で1番人気ベースライン 32-34% を下回った）。
        # 固定100イテレーションでは 32.4% まで回復するため、ランキングは固定回数を既定にする。
        callbacks = [lgb.log_evaluation(period=100)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(
                0, lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
            )
        else:
            logger.info(
                f"[{model_name}] early stopping を無効化し {n_estimators} "
                "イテレーション固定で学習します"
            )

        if val_df_for_profit is not None and val_feature_cols_for_profit is not None:
            callbacks.append(
                self._make_profit_callback(
                    val_df_for_profit, val_feature_cols_for_profit, model_name=model_name
                )
            )
            logger.info("利益コールバック有効（100iterごとに回収率をログ出力）")

        model = lgb.train(
            params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.models[model_name] = model

        logger.info(f"ランキングモデル学習完了: {model_name}")
        logger.info(f"最良イテレーション（NDCG）: {model.best_iteration}")
        logger.info(f"最良スコア（NDCG）: {model.best_score}")
        if self.ranking_best_profit_iteration is not None:
            logger.info(
                f"最良イテレーション（回収率）: {self.ranking_best_profit_iteration} "
                f"({self.ranking_best_profit_recovery_rate:.1f}%)"
            )
        return model
