"""複勝モデル戦略の予測モデル（5シード LightGBM + isotonic 校正）。

検証内容は ``docs/report/anaba_strategy_place_report/anaba_place_band_report.md``。

    # 再学習（レースが終わったらこれを実行する）
    python -m src.simulator.place_model --start-period 202401 --end-period 202607

モデル構成（レポート §1.4・§7.1）:

======================  =====================================================
ラベル                  ``placed``: 7頭以下は2着まで・8頭以上は3着まで
特徴量                  単勝版と同一の22列（:data:`src.simulator.features.FEATURE_COLS`）
学習器                  LightGBM binary を5シード独立に学習してアンサンブル
early stopping / 校正   学習期間の末尾20%（日付順）のホールドアウトのみで行う
出力                    5シードそれぞれ校正した確率の平均
======================  =====================================================

保存先（``data/model/place/``）::

    place_model_seed{0..4}.txt        LightGBM 本体
    place_calibrator_seed{0..4}.npz   isotonic 校正器（閾値の配列。pickle を使わない）
    place_model_meta.json             学習条件（特徴量の列順・学習期間など）

IMPORTANT (ラベルは着順と出走頭数から作る):
    レポートは確定払戻 ``payout_place > 0`` をラベルにしたが、このリポジトリには
    払戻データが無い。よって JRA の規定（7頭以下は2着まで・8頭以上は3着まで）を
    着順と出走頭数（取消・除外を除く）から再現する。同着は着順が同じ値になるため
    自然に入着扱いになる。

IMPORTANT (学習は 1m / 当日判断は 2m):
    単勝版と同じ既知のずれである（:mod:`src.simulator.features` の冒頭を参照）。
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.simulator.features import (
    FEATURE_COLS,
    LIVE_DECISION_SNAPSHOT,
    TRAIN_DECISION_SNAPSHOT,
    feature_frame,
)

logger = logging.getLogger(__name__)

BET_TYPE = 'place'
TARGET_COL = 'target_place'

# ``data/model`` からの相対フォルダ
DEFAULT_MODEL_SUBDIR: str = 'place'

# レポート §7.1 の学習期間の開始年月
DEFAULT_START_PERIOD: str = '202401'

MODEL_STEM = 'place_model'
CALIBRATOR_STEM = 'place_calibrator'
META_FILENAME = 'place_model_meta.json'

DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)

# 学習期間の末尾から切り出すホールドアウトの割合（日付の分位点）
HOLDOUT_FRACTION: float = 0.20

# レポートの検証（walkforward.py）と同じハイパーパラメータ
PLACE_LGBM_PARAMS: Dict[str, object] = {
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.03,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 5.0,
    'verbose': -1,
}
NUM_BOOST_ROUND: int = 3000
EARLY_STOPPING_ROUNDS: int = 100

# 8頭以上は3着まで、7頭以下は2着まで払い戻す（JRA の複勝の規定）
SMALL_FIELD_MAX_STARTERS: int = 7


# ---------------------------------------------------------------------------
# ラベル
# ---------------------------------------------------------------------------


def place_label(finish_position: pd.Series, n_starters: pd.Series) -> pd.Series:
    """着順と出走頭数から複勝の入着ラベルを作る。

    Args:
        finish_position: 着順（1始まり。取消・除外は事前に除いておくこと）
        n_starters: そのレースの出走頭数

    Returns:
        pd.Series: 入着なら 1、そうでなければ 0（int64）
    """
    paid_places = np.where(n_starters <= SMALL_FIELD_MAX_STARTERS, 2, 3)
    placed = (finish_position >= 1) & (finish_position <= paid_places)
    return placed.astype('int64')


# ---------------------------------------------------------------------------
# 保存・読込
# ---------------------------------------------------------------------------


@dataclass
class PlaceModelMeta:
    """予測側がモデルの前提を検証するためのパラメータ。"""

    feature_cols: List[str] = field(default_factory=lambda: list(FEATURE_COLS))
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    train_decision_snapshot: str = TRAIN_DECISION_SNAPSHOT
    label: str = 'placed (<=7頭: 2着まで / >=8頭: 3着まで)'
    train_start: str = ''
    train_end: str = ''
    holdout_start: str = ''
    n_train_rows: int = 0
    n_train_races: int = 0
    place_rate: float = float('nan')
    holdout_auc: List[float] = field(default_factory=list)
    num_trees: List[int] = field(default_factory=list)


def resolve_model_dir(config: dict, model_subdir: Optional[str] = None) -> Path:
    """``data/model/<subdir>`` を返す。"""
    model_dir = Path(config.get('model', {}).get('model_dir', 'data/model'))
    subdir = (
        model_subdir
        or (config.get('pipeline', {}) or {}).get('place_model_dir')
        or DEFAULT_MODEL_SUBDIR
    )
    return model_dir / subdir


def _model_path(model_dir: Path, seed: int) -> Path:
    return Path(model_dir) / f'{MODEL_STEM}_seed{seed}.txt'


def _calibrator_path(model_dir: Path, seed: int) -> Path:
    return Path(model_dir) / f'{CALIBRATOR_STEM}_seed{seed}.npz'


def load_meta(model_dir: Path) -> PlaceModelMeta:
    """運用パラメータを読み、特徴量定義との一致とファイルの揃いを確認する。

    Raises:
        FileNotFoundError: メタ・モデル・校正器のいずれかが無い場合
        ValueError: 特徴量カラムが現在の定義と食い違う場合
    """
    model_dir = Path(model_dir)
    path = model_dir / META_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f'複勝モデルが見つかりません: {path}。'
            ' 先に python -m src.simulator.place_model で学習してください。'
        )
    raw = json.loads(path.read_text(encoding='utf-8'))
    known = set(PlaceModelMeta.__dataclass_fields__)
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f'複勝モデルのメタ {path} に未知の項目があります: {unknown}')
    meta = PlaceModelMeta(**raw)
    if list(meta.feature_cols) != list(FEATURE_COLS):
        raise ValueError(
            '複勝モデルの特徴量カラムが現在の定義と一致しません。\n'
            f'  モデル: {meta.feature_cols}\n'
            f'  現在:   {list(FEATURE_COLS)}\n'
            ' src.simulator.features を変更した場合は再学習が必要です。'
        )
    if not meta.seeds:
        raise ValueError(f'複勝モデルのメタ {path} にシードがありません')
    for seed in meta.seeds:
        for required in (_model_path(model_dir, seed), _calibrator_path(model_dir, seed)):
            if not required.exists():
                # 校正器が無いと EV の前提（確率＝複勝率）が壊れるので止める
                raise FileNotFoundError(
                    f'複勝モデルのファイルが欠けています: {required}。再学習してください。'
                )
    return meta


class PlaceModel:
    """5シードのアンサンブル。各シードを校正してから平均する。"""

    def __init__(self, boosters: Sequence, calibrators: Sequence[Tuple[np.ndarray, np.ndarray]],
                 meta: PlaceModelMeta) -> None:
        self.boosters = list(boosters)
        self.calibrators = list(calibrators)
        self.meta = meta

    @classmethod
    def load(cls, model_dir: Path) -> 'PlaceModel':
        import lightgbm as lgb

        meta = load_meta(model_dir)
        boosters = [lgb.Booster(model_file=str(_model_path(model_dir, s))) for s in meta.seeds]
        calibrators = []
        for seed in meta.seeds:
            with np.load(_calibrator_path(model_dir, seed)) as data:
                calibrators.append((data['x'].copy(), data['y'].copy()))
        logger.info(
            '複勝モデル: %d シード / 学習 %s〜%s（%s レース）/ holdout AUC %s / 学習時点 %s',
            len(boosters), meta.train_start, meta.train_end, f'{meta.n_train_races:,}',
            ', '.join(f'{v:.4f}' for v in meta.holdout_auc), meta.train_decision_snapshot,
        )
        if meta.train_decision_snapshot != LIVE_DECISION_SNAPSHOT:
            logger.warning(
                '複勝モデルは %s 時点で学習し、当日は %s 時点で判断します'
                '（過去データに %s が無いための既知のずれ）。',
                meta.train_decision_snapshot, LIVE_DECISION_SNAPSHOT, LIVE_DECISION_SNAPSHOT,
            )
        return cls(boosters, calibrators, meta)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """校正済み複勝率（5シード平均）を返す。"""
        X = feature_frame(df)
        preds = []
        for booster, (x, y) in zip(self.boosters, self.calibrators):
            raw = booster.predict(X, num_iteration=booster.best_iteration or None)
            preds.append(calibrate(raw, x, y))
        return np.mean(preds, axis=0)


def calibrate(raw: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """isotonic の閾値で確率を写す（範囲外は端の値。``out_of_bounds='clip'`` と同じ）。"""
    return np.interp(np.asarray(raw, dtype='float64'), x, y)


# ---------------------------------------------------------------------------
# 学習
# ---------------------------------------------------------------------------


@dataclass
class PlaceRetrainResult:
    """再学習の結果サマリ。"""

    model_dir: str
    n_rows: int
    n_races: int
    n_fit_rows: int
    n_holdout_rows: int
    train_start: str
    train_end: str
    holdout_start: str
    place_rate: float
    holdout_auc: List[float]
    ensemble_auc: float
    num_trees: List[int]


def load_place_training_data(
    config: dict, start_period: str, end_period: str
) -> pd.DataFrame:
    """単勝版と同じ特徴量に、複勝ラベルを付けた学習データを作る。"""
    from src.simulator.retrain import N_STARTERS_COL, load_training_data

    work = load_training_data(config, start_period, end_period)
    missing = int(work[N_STARTERS_COL].isna().sum())
    if missing:
        raise ValueError(f'出走頭数が分からない行が {missing} 行あります（補完せず停止します）')
    work[TARGET_COL] = place_label(work['finish_position'], work[N_STARTERS_COL])
    logger.warning('複勝率（ラベル）: %.2f%%', float(work[TARGET_COL].mean() * 100))
    return work


def split_fit_holdout(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """学習期間の末尾20%（日付の分位点）をホールドアウトにする（レポートと同じ切り方）。"""
    cut = df['date'].quantile(1.0 - HOLDOUT_FRACTION)
    fit_df = df[df['date'] <= cut].reset_index(drop=True)
    hold_df = df[df['date'] > cut].reset_index(drop=True)
    if not len(fit_df) or not len(hold_df):
        raise ValueError(
            f'ホールドアウト分割に失敗しました: fit={len(fit_df)} 行 / holdout={len(hold_df)} 行。'
            ' 学習期間を広げてください。'
        )
    holdout_start = hold_df['date'].min()
    logger.warning(
        '[複勝] 学習 %s 行（〜%s） / ホールドアウト %s 行（%s〜）',
        f'{len(fit_df):,}', cut.date(), f'{len(hold_df):,}', holdout_start.date(),
    )
    return fit_df, hold_df, holdout_start


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def train_place_model(
    work: pd.DataFrame,
    model_dir: Path,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> PlaceRetrainResult:
    """学習データからアンサンブルを学習し、``model_dir`` に保存する。

    既存モデルは全シードの学習が成功してから上書きする（途中で失敗しても
    前日のモデルが半端に壊れない）。
    """
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression

    fit_df, hold_df, holdout_start = split_fit_holdout(work)
    X_fit, y_fit = feature_frame(fit_df), fit_df[TARGET_COL].to_numpy()
    X_hold, y_hold = feature_frame(hold_df), hold_df[TARGET_COL].to_numpy()

    boosters, calibrators, aucs, trees, hold_preds = [], [], [], [], []
    for seed in seeds:
        params = dict(PLACE_LGBM_PARAMS, seed=int(seed))
        dtrain = lgb.Dataset(X_fit, label=y_fit)
        dvalid = lgb.Dataset(X_hold, label=y_hold, reference=dtrain)
        booster = lgb.train(
            params, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        raw = booster.predict(X_hold, num_iteration=booster.best_iteration)
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(raw, y_hold)
        x, y = iso.X_thresholds_, iso.y_thresholds_
        boosters.append(booster)
        calibrators.append((x, y))
        aucs.append(_auc(y_hold, raw))
        trees.append(int(booster.best_iteration or booster.num_trees()))
        hold_preds.append(calibrate(raw, x, y))
        logger.warning(
            '[複勝] seed %d: 木 %d 本 / holdout AUC %.4f', seed, trees[-1], aucs[-1]
        )
    ensemble = np.mean(hold_preds, axis=0)
    ensemble_auc = _auc(y_hold, ensemble)
    logger.warning(
        '[複勝] アンサンブル holdout AUC %.4f（予測平均 %.2f%% / 実測 %.2f%%）',
        ensemble_auc, float(ensemble.mean() * 100), float(y_hold.mean() * 100),
    )
    if ensemble_auc < 0.75:
        logger.warning(
            '[複勝] holdout AUC が %.4f と低いです（レポートの検証では 0.81〜0.84）。'
            ' スナップショットの取得状況を確認してください。', ensemble_auc,
        )

    meta = PlaceModelMeta(
        seeds=[int(s) for s in seeds],
        train_start=str(work['date'].min().date()),
        train_end=str(work['date'].max().date()),
        holdout_start=str(holdout_start.date()),
        n_train_rows=len(work),
        n_train_races=int(work['race_id'].nunique()),
        place_rate=float(work[TARGET_COL].mean()),
        holdout_auc=aucs,
        num_trees=trees,
    )
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed, booster, (x, y) in zip(seeds, boosters, calibrators):
        booster.save_model(str(_model_path(model_dir, seed)), num_iteration=booster.best_iteration)
        np.savez(_calibrator_path(model_dir, seed), x=x, y=y)
    (model_dir / META_FILENAME).write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding='utf-8'
    )
    logger.warning('[複勝] モデルを保存: %s', model_dir)

    return PlaceRetrainResult(
        model_dir=str(model_dir),
        n_rows=len(work),
        n_races=meta.n_train_races,
        n_fit_rows=len(fit_df),
        n_holdout_rows=len(hold_df),
        train_start=meta.train_start,
        train_end=meta.train_end,
        holdout_start=meta.holdout_start,
        place_rate=meta.place_rate,
        holdout_auc=aucs,
        ensemble_auc=ensemble_auc,
        num_trees=trees,
    )


def run_place_retrain(
    config: dict,
    *,
    start_period: str,
    end_period: str,
    model_subdir: Optional[str] = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> PlaceRetrainResult:
    """学習データを作って学習し、``data/model/place`` に保存する。"""
    work = load_place_training_data(config, start_period, end_period)
    return train_place_model(work, resolve_model_dir(config, model_subdir), seeds=seeds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='複勝モデル戦略のモデルを学習する')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='設定ファイルパス')
    parser.add_argument('--start-period', type=str, default=DEFAULT_START_PERIOD,
                        help=f'学習開始年月 (YYYYMM)。既定 {DEFAULT_START_PERIOD}（レポートの指定期間）')
    parser.add_argument('--end-period', type=str, required=True,
                        help='学習終了年月 (YYYYMM)')
    parser.add_argument('--model-dir', type=str, default=None,
                        help=f'data/model 配下の保存フォルダ（既定 {DEFAULT_MODEL_SUBDIR}）')
    return parser.parse_args()


def main() -> None:
    """CLI エントリポイント。"""
    from src.cli_common import build_config

    logging.basicConfig(
        level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s'
    )
    args = _parse_args()
    config = build_config(args.config)
    result = run_place_retrain(
        config,
        start_period=args.start_period,
        end_period=args.end_period,
        model_subdir=args.model_dir,
    )
    print()
    print('=== 複勝モデル 学習完了 ===')
    print(f'  保存先       : {result.model_dir}')
    print(f'  学習期間     : {result.train_start} 〜 {result.train_end}'
          f'（{result.n_rows:,} 行 / {result.n_races:,} レース / 複勝率 {result.place_rate:.2%}）')
    print(f'  fit / holdout: {result.n_fit_rows:,} 行 / {result.n_holdout_rows:,} 行'
          f'（{result.holdout_start}〜）')
    print(f'  木の本数     : {result.num_trees}')
    print(f'  holdout AUC  : {", ".join(f"{v:.4f}" for v in result.holdout_auc)}'
          f'（アンサンブル {result.ensemble_auc:.4f}）')


if __name__ == '__main__':
    main()
