"""レース後にモデルを再学習する単独実行プログラム（spec §4）。

過去の締切前オッズ（10m / 5m）から特徴量2列を作り、確定着順をラベルにして
LightGBM を学習する。あわせてレース選別のしきい値（``pool_5m`` の q0.75）を
**学習期間の分布から**求め、モデルと対で保存する。

    # 全期間で再学習（レースが終わったらこれを実行する）
    python -m src.simulator.retrain --start-period 202401 --end-period 202607

    # しきい値の分位点を変える / ホールドアウト幅を変える
    python -m src.simulator.retrain --start-period 202401 --end-period 202607 \
        --pool-quantile 0.75 --holdout-months 6

出力:

==========================================================  ====================
``data/model/simulator/win_pool_filter.txt``                学習済み Booster
``data/model/simulator/win_pool_filter_calibrator.pkl``     isotonic 校正器
``data/model/simulator/win_pool_filter_simulator.json``     しきい値等の運用パラメータ
==========================================================  ====================

IMPORTANT (時系列リーク防止):
    - 学習期間の末尾 ``--holdout-months`` ヶ月を holdout として切り出し、
      early stopping と isotonic キャリブレーションはそこだけで行う。
      木の学習（fit）期間には触れさせない。
    - ``pool_threshold`` も**学習期間の分布**から決める。予測時に当日の
      レース群から取り直すと未来を見ない前提が崩れる（spec §5.2）。
    - 特徴量は 10m / 5m の締切前スナップショットのみ。確定オッズ・確定票数・
      払戻は使わない。着順はラベルとしてのみ使う。

IMPORTANT (木を小さくする理由 / spec §4):
    config 既定（``num_leaves: 113`` / ``min_data_in_leaf: 100``）は2列に
    対して木が大きすぎ、1本目で holdout AUC が飽和して early stopping が
    即発動し ``num_trees=1`` になる（片方の特徴量の importance が 0 になる）。
    よって本モジュールは ``SPEC_LGBM_PARAMS`` で小さい木に上書きする。
"""
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.model_creator import ModelCreator
from src.simulator.artifacts import (
    DEFAULT_MODEL_FILENAME,
    SimulatorParams,
    resolve_model_path,
    save_params,
)
from src.simulator.dataset import load_finish_positions
from src.simulator.odds_series_loader import load_odds_series
from src.simulator.features import (
    DECISION_SNAPSHOT,
    FEATURE_COLS,
    POOL_COL,
    REQUIRED_SNAPSHOTS,
    attach_features,
    describe_pool_quantiles,
    drop_incomplete_rows,
    feature_frame,
    pool_threshold,
    race_pool,
)

logger = logging.getLogger(__name__)

BET_TYPE = 'win'
TARGET_COL = 'target_win'

# spec §5.2 で採用している分位点
DEFAULT_POOL_QUANTILE: float = 0.75

# 学習期間の末尾から切り出す holdout の月数（early stopping / 校正に使う）
DEFAULT_HOLDOUT_MONTHS: int = 6

# spec §6 の期待値しきい値。config の ``min_ev_win``（既定 1.05）は本体
# パイプライン向けに調整された別物なので、本条件の値をここで持つ。
DEFAULT_MIN_EV: float = 1.10

# spec §4 のハイパーパラメータ。2列モデルには小さい木が要る。
SPEC_LGBM_PARAMS: Dict[str, object] = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'max_depth': 4,
    'learning_rate': 0.03,
    'min_data_in_leaf': 200,
    'feature_fraction': 1.0,
    'bagging_fraction': 0.9,
    'bagging_freq': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 1.0,
    'verbose': -1,
    'seed': 42,
}


@dataclass
class RetrainResult:
    """再学習の結果サマリ。"""

    model_path: str
    pool_threshold: float
    pool_quantile: float
    n_rows: int
    n_races: int
    n_fit_rows: int
    n_holdout_rows: int
    train_start: str
    train_end: str
    holdout_start: str
    holdout_auc: float
    num_trees: int
    importance: List[Tuple[str, float]]


def load_training_data(
    config: dict, start_period: str, end_period: str
) -> pd.DataFrame:
    """締切前オッズと確定着順を結合した学習データを作る。

    Args:
        config: 全体設定辞書
        start_period: 開始年月 (YYYYMM)
        end_period: 終了年月 (YYYYMM)

    Returns:
        pd.DataFrame: 特徴量・``pool_5m``・``target_win``・``date`` を持つ表

    Raises:
        ValueError: 着順が結合できない行がある場合（補完せず停止する）
    """
    data_cfg = config.get('data', {})
    odds_series_dir = Path(
        data_cfg.get('odds_series_output_dir', 'data/processed/odds_series')
    )
    raw = load_odds_series(
        odds_series_dir, start_period, end_period, snapshots=REQUIRED_SNAPSHOTS
    )
    work = attach_features(raw)
    work = drop_incomplete_rows(work, context='学習')

    # 着順（ラベル）を結合する。race_id は horses 側が int64 なので揃える。
    work['race_id'] = work['race_id'].astype('int64')
    parsed_data_dir = Path(data_cfg.get('parsed_data_dir', 'data/processed'))
    horses = load_finish_positions(parsed_data_dir, work['race_id'])
    work = work.merge(horses, on=['race_id', 'horse_number'], how='left')

    missing = int(work['finish_position'].isna().sum())
    if missing:
        raise ValueError(
            f'着順が結合できない行が {missing} 行あります。'
            f' data/processed/horses が期間 {start_period}〜{end_period} を'
            ' 覆っているか確認してください（補完せず停止します）。'
        )
    work['finish_position'] = work['finish_position'].astype('int64')

    # 出走取消・除外・失格は学習対象外
    scratched = int((work['finish_position'] <= 0).sum())
    if scratched:
        logger.info('出走取消・除外・失格を除外: %s 行', f'{scratched:,}')
        work = work[work['finish_position'] > 0].reset_index(drop=True)

    work[TARGET_COL] = (work['finish_position'] == 1).astype('int64')
    work['date'] = pd.to_datetime(work['date'])
    work = work.sort_values(['date', 'race_id', 'horse_number']).reset_index(drop=True)

    logger.warning(
        '学習データ: %s 行 / %s レース / %s 〜 %s（勝率 %.2f%%）',
        f'{len(work):,}', f'{work["race_id"].nunique():,}',
        work['date'].min().date(), work['date'].max().date(),
        float(work[TARGET_COL].mean() * 100),
    )
    return work


def split_fit_holdout(
    df: pd.DataFrame, holdout_months: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """学習期間の末尾を holdout として時系列分割する。

    Args:
        df: ``date`` 昇順の学習データ
        holdout_months: 末尾から切り出す月数

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
            (fit, holdout, 分割日)

    Raises:
        ValueError: どちらかが空になる場合
    """
    split_date = df['date'].max() - pd.DateOffset(months=holdout_months)
    fit_df = df[df['date'] < split_date].reset_index(drop=True)
    holdout_df = df[df['date'] >= split_date].reset_index(drop=True)
    if not len(fit_df) or not len(holdout_df):
        raise ValueError(
            f'ホールドアウト分割に失敗しました（holdout_months={holdout_months}）: '
            f'fit={len(fit_df)} 行 / holdout={len(holdout_df)} 行。'
            ' 学習期間を広げるか --holdout-months を小さくしてください。'
        )
    logger.warning(
        '学習 %s 行（〜%s） / ホールドアウト %s 行（%s〜）',
        f'{len(fit_df):,}', split_date.date(),
        f'{len(holdout_df):,}', split_date.date(),
    )
    return fit_df, holdout_df, split_date


def _sample_weight(config: dict, creator: ModelCreator, df: pd.DataFrame) -> np.ndarray:
    """時間減衰サンプルウェイトを計算する（spec §4）。

    Args:
        config: 全体設定辞書
        creator: ModelCreator
        df: 学習データ（``date`` 列必須）

    Returns:
        np.ndarray: サンプルウェイト
    """
    decay_cfg = config.get('model', {}).get('time_decay', {})
    return creator.compute_time_decay_weights(
        df['date'].reset_index(drop=True),
        recent_years=float(decay_cfg.get('recent_years', 2.0)),
        mid_years=float(decay_cfg.get('mid_years', 4.0)),
        w_recent=float(decay_cfg.get('w_recent', 3.0)),
        w_mid=float(decay_cfg.get('w_mid', 2.0)),
    )


def run_retrain(
    config: dict,
    *,
    start_period: str,
    end_period: str,
    pool_quantile: float = DEFAULT_POOL_QUANTILE,
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    min_ev: Optional[float] = None,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    use_spec_params: bool = True,
) -> RetrainResult:
    """再学習を実行し、モデル・キャリブレータ・運用パラメータを保存する。

    Args:
        config: 全体設定辞書
        start_period: 学習開始年月 (YYYYMM)
        end_period: 学習終了年月 (YYYYMM)
        pool_quantile: ``pool_5m`` しきい値を取る分位点
        holdout_months: 学習期間末尾から切り出す holdout の月数
        min_ev: 予測側の既定 EV しきい値（省略時は ``DEFAULT_MIN_EV``）
        model_filename: ``data/model`` 配下の保存ファイル名
        use_spec_params: True なら spec §4 のハイパーパラメータで上書きする。
            False なら config の ``model.lightgbm`` をそのまま使う

    Returns:
        RetrainResult: 学習結果のサマリ
    """
    work = load_training_data(config, start_period, end_period)

    # --- レース選別しきい値を学習期間の分布から決める（spec §5.2） ---
    pool_by_race = race_pool(work)
    threshold = pool_threshold(pool_by_race, pool_quantile)
    logger.warning('学習期間のレース総投票額（%s時点）の分位点:', DECISION_SNAPSHOT)
    for label, value in describe_pool_quantiles(pool_by_race):
        mark = ' ← 採用' if abs(value - threshold) < 1e-6 else ''
        logger.warning('  %s: %15s 円%s', label, f'{value:,.0f}', mark)

    fit_df, holdout_df, split_date = split_fit_holdout(work, holdout_months)

    creator = ModelCreator(config, config.get('features_def'))
    if use_spec_params:
        # 2列モデルに config 既定の大きい木を使うと1本目で飽和する（spec §4）
        creator.update_lgbm_params('lightgbm', SPEC_LGBM_PARAMS)
        logger.warning(
            'spec §4 のハイパーパラメータで学習します（num_leaves=%s / '
            'max_depth=%s / min_data_in_leaf=%s）',
            SPEC_LGBM_PARAMS['num_leaves'], SPEC_LGBM_PARAMS['max_depth'],
            SPEC_LGBM_PARAMS['min_data_in_leaf'],
        )

    X_fit, y_fit = feature_frame(fit_df), fit_df[TARGET_COL]
    X_hold, y_hold = feature_frame(holdout_df), holdout_df[TARGET_COL]

    creator.train(
        X_fit, y_fit, X_hold, y_hold,
        model_name=BET_TYPE,
        sample_weight=_sample_weight(config, creator, fit_df),
    )

    # キャリブレータは holdout（木の学習に使っていない期間）で fit する
    cal_cfg = config.get('model', {}).get('calibration', {})
    creator.calibrate_model(
        BET_TYPE, X_hold, y_hold,
        method=cal_cfg.get('method', 'isotonic'),
        min_samples_per_bin=int(cal_cfg.get('min_samples_per_bin', 0)),
        min_pos_per_bin=int(cal_cfg.get('min_pos_per_bin', 0)),
    )

    hold_proba = creator.predict(X_hold, model_name=BET_TYPE)
    hold_metrics = creator.evaluate_model(y_hold, hold_proba, threshold=0.5)
    holdout_auc = float(hold_metrics.get('auc', float('nan')))

    booster = creator.models[BET_TYPE]
    num_trees = int(booster.best_iteration or booster.num_trees())
    importance = [
        (str(row.feature), float(row.importance))
        for row in creator.get_feature_importance(BET_TYPE, top_n=len(FEATURE_COLS)).itertuples()
    ]
    logger.warning(
        '学習完了: 木 %d 本 / holdout AUC %.4f / importance %s',
        num_trees, holdout_auc,
        ', '.join(f'{name}: {value:,.0f}' for name, value in importance),
    )
    if num_trees <= 1:
        logger.warning(
            '木が %d 本しか育っていません。1本目で holdout AUC が飽和した'
            ' 可能性があります（spec §4）。--no-spec-params を使っている場合は'
            ' 外して再実行してください。', num_trees,
        )

    # LightGBM は保存先ディレクトリを自動生成しないため先に作る
    model_path = resolve_model_path(creator.model_dir, model_filename)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    creator.save_model(BET_TYPE, model_filename)

    params = SimulatorParams(
        pool_threshold=threshold,
        pool_quantile=float(pool_quantile),
        feature_cols=list(FEATURE_COLS),
        min_ev=float(min_ev if min_ev is not None else DEFAULT_MIN_EV),
        odds_snapshot=DECISION_SNAPSHOT,
        train_start=str(work['date'].min().date()),
        train_end=str(work['date'].max().date()),
        n_train_rows=len(work),
        n_train_races=int(work['race_id'].nunique()),
        holdout_auc=holdout_auc,
        num_trees=num_trees,
    )
    save_params(params, model_path)

    return RetrainResult(
        model_path=str(model_path),
        pool_threshold=threshold,
        pool_quantile=float(pool_quantile),
        n_rows=len(work),
        n_races=int(work['race_id'].nunique()),
        n_fit_rows=len(fit_df),
        n_holdout_rows=len(holdout_df),
        train_start=str(work['date'].min().date()),
        train_end=str(work['date'].max().date()),
        holdout_start=str(split_date.date()),
        holdout_auc=holdout_auc,
        num_trees=num_trees,
        importance=importance,
    )


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数
    """
    parser = argparse.ArgumentParser(
        description='当日オッズ・投票数モデルを再学習する（spec §4）'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='設定ファイルパス')
    parser.add_argument('--start-period', type=str, required=True,
                        help='学習開始年月 (YYYYMM)')
    parser.add_argument('--end-period', type=str, required=True,
                        help='学習終了年月 (YYYYMM)')
    parser.add_argument('--pool-quantile', type=float,
                        default=DEFAULT_POOL_QUANTILE,
                        help=f'{POOL_COL} しきい値を取る分位点（既定 0.75）')
    parser.add_argument('--holdout-months', type=int,
                        default=DEFAULT_HOLDOUT_MONTHS,
                        help='学習期間末尾から切り出す holdout の月数')
    parser.add_argument('--min-ev', type=float, default=None,
                        help=f'予測側の既定 EV しきい値（既定 {DEFAULT_MIN_EV}）')
    parser.add_argument('--model-filename', type=str,
                        default=DEFAULT_MODEL_FILENAME,
                        help='data/model 配下の保存ファイル名')
    parser.add_argument('--no-spec-params', action='store_true',
                        help='spec §4 のHPで上書きせず config の設定を使う')
    return parser.parse_args()


def main() -> None:
    """CLI エントリポイント。"""
    from src.cli_common import build_config

    logging.basicConfig(
        level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s'
    )
    args = _parse_args()
    config = build_config(args.config)

    result = run_retrain(
        config,
        start_period=args.start_period,
        end_period=args.end_period,
        pool_quantile=args.pool_quantile,
        holdout_months=args.holdout_months,
        min_ev=args.min_ev,
        model_filename=args.model_filename,
        use_spec_params=not args.no_spec_params,
    )

    print()
    print('=== 再学習完了 ===')
    print(f'  モデル       : {result.model_path}')
    print(f'  学習期間     : {result.train_start} 〜 {result.train_end}'
          f'（{result.n_rows:,} 行 / {result.n_races:,} レース）')
    print(f'  fit / holdout: {result.n_fit_rows:,} 行 / '
          f'{result.n_holdout_rows:,} 行（{result.holdout_start}〜）')
    print(f'  木の本数     : {result.num_trees}')
    print(f'  holdout AUC  : {result.holdout_auc:.4f}')
    print(f'  pool しきい値: {result.pool_threshold:,.0f} 円'
          f'（q{result.pool_quantile:.2f}）')
    print('  importance   : ' + ', '.join(
        f'{name} {value:,.0f}' for name, value in result.importance
    ))
    print()
    print('当日は python -m src.simulator.predict_today で予測できます。')


if __name__ == '__main__':
    main()
