"""穴馬戦略のモデルを再学習する単独実行プログラム。

過去の締切前オッズ（60m/30m/10m/5m/1m）から22列の特徴量を作り、確定着順を
ラベルにして LightGBM を学習し、isotonic 回帰で確率を校正する。

    # 全期間で再学習（レースが終わったらこれを実行する）
    python -m src.simulator.retrain --start-period 202301 --end-period 202607

    # ホールドアウト幅を変える
    python -m src.simulator.retrain --start-period 202301 --end-period 202607 \
        --holdout-months 6

出力:

==========================================================  ====================
``data/model/win/win_anaba.txt``                      学習済み Booster
``data/model/win/win_anaba_calibrator.pkl``           isotonic 校正器
``data/model/win/win_anaba_simulator.json``           運用パラメータ
==========================================================  ====================

IMPORTANT (学習は 1m / 当日判断は 2m):
    過去データに 2m スナップショットが存在しないため、学習の「締切直前」は
    ``1m`` を使う。当日の判断は指示どおり ``2m`` で行う。この時点のずれは
    :mod:`src.simulator.features` の冒頭に記録した既知の制約である。

IMPORTANT (学習期間は 2023年1月以降に限る):
    ``1m`` スナップショットが実質的に更新されるのは 2023年以降である。
    2016〜2021年は 98% のレースで ``5m`` と ``1m`` の総票数が同一であり、
    この区間の特徴量を算出できない。2022年以前を混ぜると「年代の代理変数」
    になるため、既定の開始年月を ``202301`` とし、それより前を指定した
    場合は警告する。

IMPORTANT (時系列リーク防止):
    - 学習期間の末尾 ``--holdout-months`` ヶ月を holdout として切り出し、
      early stopping と isotonic キャリブレーションはそこだけで行う。
      木の学習（fit）期間には触れさせない。
    - 特徴量は締切前スナップショットのみ。確定オッズ・確定票数・払戻は
      使わない。着順はラベルとしてのみ使う。

NOTE (確率が校正されていることが前提):
    EV = 予測確率 × オッズ という判定は、予測確率がそのまま勝率として
    読めることを前提にしている。よってキャリブレータが無いまま予測しては
    ならない（:func:`src.simulator.artifacts.require_model` が止める）。
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
from src.simulator.features import (
    DRIFT_BASE_SNAPSHOT,
    FEATURE_COLS,
    POOL_COL,
    TRAIN_DECISION_SNAPSHOT,
    build_features,
    describe_pool_quantiles,
    drop_incomplete_rows,
    feature_frame,
    normalize_late_columns,
    race_pool,
    required_snapshots,
)
from src.simulator.odds_series_loader import load_odds_series
from src.simulator.strategy import AnabaRule

logger = logging.getLogger(__name__)

BET_TYPE = 'win'
TARGET_COL = 'target_win'

# 出走頭数（取消・除外を除く）。複勝モデルのラベル作成に使う（特徴量ではない）。
N_STARTERS_COL = 'n_starters'

# ``1m`` が実質的に更新される最初の年月。これより前は学習に使えない。
MIN_RELIABLE_PERIOD: str = '202301'

# 学習期間の末尾から切り出す holdout の月数（early stopping / 校正に使う）
DEFAULT_HOLDOUT_MONTHS: int = 6

# レポート §2.2 のハイパーパラメータ（29列・11フォールドで AUC 0.834〜0.852）
ANABA_LGBM_PARAMS: Dict[str, object] = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'max_depth': 5,
    'learning_rate': 0.03,
    'min_data_in_leaf': 200,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.0,
    'lambda_l2': 5.0,
    'verbose': -1,
    'seed': 42,
}


@dataclass
class RetrainResult:
    """再学習の結果サマリ。"""

    model_path: str
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
    pool_quantiles: List[Tuple[str, float]]


def load_training_data(
    config: dict, start_period: str, end_period: str
) -> pd.DataFrame:
    """締切前オッズと確定着順を結合した学習データを作る。

    Args:
        config: 全体設定辞書
        start_period: 開始年月 (YYYYMM)
        end_period: 終了年月 (YYYYMM)

    Returns:
        pd.DataFrame: 22列の特徴量・``POOL_COL``・``target_win``・``date``

    Raises:
        ValueError: 着順が結合できない行がある場合（補完せず停止する）
    """
    if start_period < MIN_RELIABLE_PERIOD:
        logger.warning(
            '開始年月 %s は %s より前です。%s スナップショットは 2022年以前'
            ' ほぼ更新が取れておらず、混ぜると「年代の代理変数」になります。'
            ' --start-period %s 以降を推奨します。',
            start_period, MIN_RELIABLE_PERIOD, TRAIN_DECISION_SNAPSHOT,
            MIN_RELIABLE_PERIOD,
        )

    data_cfg = config.get('data', {})
    odds_series_dir = Path(
        data_cfg.get('odds_series_output_dir', 'data/processed/odds_series')
    )
    snaps = required_snapshots(TRAIN_DECISION_SNAPSHOT)
    raw = load_odds_series(
        odds_series_dir, start_period, end_period, snapshots=snaps
    )
    # 学習の「締切直前」は 1m。特徴量コードは late という名前で受け取る。
    raw = normalize_late_columns(raw, TRAIN_DECISION_SNAPSHOT)
    work = build_features(raw)
    work = drop_incomplete_rows(work, context='学習')

    # 着順（ラベル）を結合する。race_id は horses 側が int64 なので揃える。
    work['race_id'] = work['race_id'].astype('int64')
    parsed_data_dir = Path(data_cfg.get('parsed_data_dir', 'data/processed'))
    horses = load_finish_positions(parsed_data_dir, work['race_id'])
    # 出走頭数（取消・除外を除く）。複勝の払戻対象（7頭以下は2着まで）の判定に使う。
    starters = (
        horses[horses['finish_position'] > 0].groupby('race_id').size()
        .rename(N_STARTERS_COL)
    )
    work = work.merge(horses, on=['race_id', 'horse_number'], how='left')
    work = work.merge(starters, left_on='race_id', right_index=True, how='left')

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
    """時間減衰サンプルウェイトを計算する。

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
    holdout_months: int = DEFAULT_HOLDOUT_MONTHS,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    use_anaba_params: bool = True,
) -> RetrainResult:
    """再学習を実行し、モデル・キャリブレータ・運用パラメータを保存する。

    Args:
        config: 全体設定辞書
        start_period: 学習開始年月 (YYYYMM)
        end_period: 学習終了年月 (YYYYMM)
        holdout_months: 学習期間末尾から切り出す holdout の月数
        model_filename: ``data/model`` 配下の保存ファイル名
        use_anaba_params: True ならレポート §2.2 のハイパーパラメータを使う

    Returns:
        RetrainResult: 学習結果のサマリ
    """
    work = load_training_data(config, start_period, end_period)

    # レース総投票額の分布をログに残す（購入条件の5億円がどこに来るかの確認）
    pool_by_race = race_pool(work)
    quantiles = describe_pool_quantiles(pool_by_race)
    rule = AnabaRule()
    logger.warning(
        '学習期間のレース総投票額（%s時点）の分位点:', TRAIN_DECISION_SNAPSHOT
    )
    for label, value in quantiles:
        logger.warning('  %s: %15s 円', label, f'{value:,.0f}')
    passed = float((pool_by_race >= rule.min_pool_yen).mean() * 100)
    logger.warning(
        '購入条件の総投票額しきい値 %s円 を満たすレースは学習期間の %.1f%% です',
        f'{rule.min_pool_yen:,.0f}', passed,
    )

    fit_df, holdout_df, split_date = split_fit_holdout(work, holdout_months)

    creator = ModelCreator(config, config.get('features_def'))
    if use_anaba_params:
        creator.update_lgbm_params('lightgbm', ANABA_LGBM_PARAMS)
        logger.warning(
            'レポート §2.2 のハイパーパラメータで学習します'
            '（num_leaves=%s / max_depth=%s / min_data_in_leaf=%s / lambda_l2=%s）',
            ANABA_LGBM_PARAMS['num_leaves'], ANABA_LGBM_PARAMS['max_depth'],
            ANABA_LGBM_PARAMS['min_data_in_leaf'], ANABA_LGBM_PARAMS['lambda_l2'],
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
        for row in creator.get_feature_importance(
            BET_TYPE, top_n=len(FEATURE_COLS)
        ).itertuples()
    ]
    logger.warning(
        '学習完了: 木 %d 本 / holdout AUC %.4f', num_trees, holdout_auc,
    )
    logger.warning(
        'importance 上位5列: %s',
        ', '.join(f'{name}: {value:,.0f}' for name, value in importance[:5]),
    )
    if holdout_auc < 0.75:
        logger.warning(
            'holdout AUC が %.4f と低いです。レポートの検証では 0.834〜0.852'
            ' でした。スナップショットの取得状況を確認してください。',
            holdout_auc,
        )

    # LightGBM は保存先ディレクトリを自動生成しないため先に作る
    model_path = resolve_model_path(creator.model_dir, model_filename)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    creator.save_model(BET_TYPE, model_filename)

    params = SimulatorParams(
        feature_cols=list(FEATURE_COLS),
        train_decision_snapshot=TRAIN_DECISION_SNAPSHOT,
        drift_base_snapshot=DRIFT_BASE_SNAPSHOT,
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
        pool_quantiles=quantiles,
    )


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数
    """
    parser = argparse.ArgumentParser(
        description='穴馬戦略のモデルを再学習する'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='設定ファイルパス')
    parser.add_argument('--start-period', type=str, required=True,
                        help=f'学習開始年月 (YYYYMM)。{MIN_RELIABLE_PERIOD} 以降を推奨')
    parser.add_argument('--end-period', type=str, required=True,
                        help='学習終了年月 (YYYYMM)')
    parser.add_argument('--holdout-months', type=int,
                        default=DEFAULT_HOLDOUT_MONTHS,
                        help='学習期間末尾から切り出す holdout の月数')
    parser.add_argument('--model-filename', type=str,
                        default=DEFAULT_MODEL_FILENAME,
                        help='data/model 配下の保存ファイル名')
    parser.add_argument('--no-anaba-params', action='store_true',
                        help='レポートのHPで上書きせず config の設定を使う')
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
        holdout_months=args.holdout_months,
        model_filename=args.model_filename,
        use_anaba_params=not args.no_anaba_params,
    )

    print()
    print('=== 再学習完了 ===')
    print(f'  モデル       : {result.model_path}')
    print(f'  学習期間     : {result.train_start} 〜 {result.train_end}'
          f'（{result.n_rows:,} 行 / {result.n_races:,} レース）')
    print(f'  fit / holdout: {result.n_fit_rows:,} 行 / '
          f'{result.n_holdout_rows:,} 行（{result.holdout_start}〜）')
    print(f'  特徴量       : {len(FEATURE_COLS)} 列')
    print(f'  木の本数     : {result.num_trees}')
    print(f'  holdout AUC  : {result.holdout_auc:.4f}')
    print('  importance   : ' + ', '.join(
        f'{name} {value:,.0f}' for name, value in result.importance[:8]
    ))
    print()
    print('当日は python -m src.simulator.predict_today で予測できます。')


if __name__ == '__main__':
    main()
