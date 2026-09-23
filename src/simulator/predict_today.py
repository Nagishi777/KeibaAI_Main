"""当日のオッズ・投票数で予測しベット推奨を出す単独実行プログラム（spec §6）。

レース当日、10m・5m時点のオッズと総票数を取得して特徴量化し、学習済みモデルで
各馬の勝率を予測して、買うべき馬と賭け金を CSV に出力する。

    # 取得から予測まで一気通貫（当日これを実行する）
    python -m src.simulator.predict_today --fetch

    # 既に取得済みの CSV から予測だけ行う
    python -m src.simulator.predict_today

    # 日付・EVしきい値を指定する
    python -m src.simulator.predict_today --date 2026-07-18 --min-ev 1.10

処理の流れ（spec §1）::

    [1] 10m時点と5m時点のオッズ・総票数を取得（--fetch で当日スクレイパを起動）
    [2] 馬番別の投票額を逆算（オッズの定義式から）
    [3] 特徴量2列を作る: odds_5m / inc_share_10m_5m
    [4] 学習済み LightGBM で各馬の勝率を予測
    [5] 【フィルタ】5m時点のレース総額がしきい値未満のレースは丸ごと見送る
    [6] 残ったレースで EV = 予測確率 × 5mオッズ >= min_ev の馬を買う
    [7] 賭け金は fractional Kelly で決定、1レース最大 N 点

出力: ``output/simulator/{YYYYMMDD}_bets.csv``（推奨のみ）と
``{YYYYMMDD}_all.csv``（全馬の予測。見送り理由つき）。

IMPORTANT (しきい値は学習時の固定値を使う):
    ``pool_5m`` のしきい値は再学習時に**学習期間の分布**から決めた値を
    そのまま使う（spec §5.2）。当日のレース群から取り直すと「その日
    たまたま厚かったレース」を基準にすることになり、未来を見ない前提が
    崩れる。

IMPORTANT (5m時点で判断する理由):
    実際の投票締切は発走の約1分前であり、1mオッズを見てから発注する時間的
    余裕は無い（spec §8-5）。よって判断を 5m 時点に前倒ししている。
    5m以降のオッズ（4m〜10s）は取得・保存するが予測には使わない。

IMPORTANT (本条件は実運用の推奨ではない):
    単一分割・多重比較を含む調査結果であり、期間安定性は未検証。
    的中率3.85%・平均オッズ80倍の高分散戦略である（spec §7・§8）。
"""
import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.evaluation import ModelEvaluator
from src.models.model_creator import ModelCreator
from src.simulator.artifacts import (
    DEFAULT_MODEL_FILENAME,
    SimulatorParams,
    load_params,
    require_model,
    resolve_model_path,
)
from src.simulator.features import (
    INC_SHARE_COL,
    ODDS_COL,
    POOL_COL,
    attach_features,
    drop_incomplete_rows,
    feature_frame,
)
from src.simulator.realtime_loader import (
    DEFAULT_REALTIME_DIR,
    load_today_snapshots,
)

logger = logging.getLogger(__name__)

BET_TYPE = 'win'

# 推奨 CSV の出力先（``output`` 配下）
OUTPUT_SUBDIR = 'simulator'

# 見送り理由のラベル（全馬 CSV の ``skip_reason`` に入る）
REASON_BET = ''
REASON_POOL = 'pool_filter'          # レース総額がしきい値未満
REASON_EV = 'ev_below_threshold'     # EV がしきい値未満
REASON_PROBA = 'proba_below_min'     # 予測確率が下限未満
REASON_MAX_BETS = 'max_bets_per_race'  # 1レース点数上限で次点に漏れた
REASON_KELLY = 'kelly_zero'          # Kelly が 0 円（賭けない）


@dataclass
class PredictionSummary:
    """当日予測のサマリ。"""

    target_date: str
    n_races_total: int
    n_races_passed: int
    n_horses: int
    n_bets: int
    total_bet: float
    mean_odds: float
    pool_threshold: float
    min_ev: float
    bets_path: str
    all_path: str


def fetch_today_odds(
    target_date: dt.date,
    *,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    headless: bool = True,
) -> None:
    """当日スクレイパを起動して時点別オッズ CSV を作る。

    発走時刻に合わせて取得するスケジューラなので、当日の開催中に実行する。
    最終レースの取得が終わるまで動き続け、``Ctrl+C`` で停止できる。
    停止後にそのまま予測へ進む。

    Args:
        target_date: 対象日
        realtime_dir: 時点別オッズ CSV の出力先（予測側が読む場所と揃える）
        headless: 出馬表取得用ブラウザを非表示にするか

    Raises:
        RuntimeError: 当日の発走時刻を取得できなかった場合
    """
    # JV-Link / ブラウザに依存するため、--fetch 指定時のみ import する
    from src.scraping.realtime_odds import (
        get_today_schedule,
        run_scheduler,
    )

    logger.warning('当日オッズの取得を開始します: %s', target_date)
    schedule = get_today_schedule(target_date, headless=headless)
    run_scheduler(schedule, target_date, output_dir=Path(realtime_dir))


def predict_proba(
    config: dict, df: pd.DataFrame, model_filename: str
) -> Tuple[np.ndarray, SimulatorParams]:
    """学習済みモデルで各馬の勝率を予測する。

    Args:
        config: 全体設定辞書
        df: 特徴量を持つ DataFrame
        model_filename: ``data/model`` 配下のモデルファイル名

    Returns:
        Tuple[np.ndarray, SimulatorParams]: (予測確率, 運用パラメータ)
    """
    creator = ModelCreator(config, config.get('features_def'))
    model_path = resolve_model_path(creator.model_dir, model_filename)
    require_model(model_path)
    params = load_params(model_path)

    creator.load_model(model_filename, model_name=BET_TYPE)
    proba = creator.predict(feature_frame(df), model_name=BET_TYPE)
    return proba, params


def select_bets(
    config: dict,
    df: pd.DataFrame,
    proba: np.ndarray,
    params: SimulatorParams,
    *,
    min_ev: Optional[float] = None,
) -> pd.DataFrame:
    """フィルタ・EV・Kelly を適用して全馬に賭け判断を付ける（spec §6）。

    賭けない馬も落とさず ``skip_reason`` を付けて返す。当日の意思決定では
    「なぜ買わなかったか」が確認できる必要があるため。

    賭け金は ``ModelEvaluator._kelly_criterion_vec`` をそのまま使う。
    バックテストと同じ関数を通すことで、検証時と当日で賭け金の定義が
    ずれることを防ぐ。

    Args:
        config: 全体設定辞書
        df: 特徴量・``pool_5m``・``odds_5m`` を持つ DataFrame
        proba: 予測確率（df と同じ行順）
        params: 運用パラメータ（しきい値・既定 EV）
        min_ev: EV しきい値（省略時は ``params.min_ev``）

    Returns:
        pd.DataFrame: 全馬に ``pred_proba`` / ``ev`` / ``bet_amount`` /
            ``skip_reason`` を付けた表（確率降順）
    """
    evaluator = ModelEvaluator(config.get('evaluation', {}))
    threshold_ev = float(min_ev if min_ev is not None else params.min_ev)
    min_proba = evaluator.min_proba_win
    max_bets = evaluator.max_bets_per_race_win

    work = df.copy().reset_index(drop=True)
    work['pred_proba'] = np.asarray(proba, dtype='float64')
    # 賭け判断オッズは 5m 時点（spec §6「賭け値と特徴量の時点が揃っている」）
    work['odds_used'] = work[ODDS_COL].astype('float64')
    work['ev'] = work['pred_proba'] * work['odds_used']

    # [5] レース単位のフィルタ。通らないレースは丸ごと見送る。
    passed_pool = work[POOL_COL] >= params.pool_threshold
    # [6] 期待値・最低確率
    passed_ev = work['ev'] >= threshold_ev
    passed_proba = work['pred_proba'] >= min_proba

    reason = pd.Series(REASON_BET, index=work.index, dtype='object')
    reason[~passed_proba] = REASON_PROBA
    reason[~passed_ev] = REASON_EV
    reason[~passed_pool] = REASON_POOL
    work['skip_reason'] = reason

    # [7] 1レース最大点数。確率の高い順に残す（extract_bet_rows と同じ基準）。
    candidate = work['skip_reason'] == REASON_BET
    if max_bets > 0 and candidate.any():
        kept = (
            work[candidate]
            .sort_values('pred_proba', ascending=False)
            .groupby('race_id', group_keys=False)
            .head(max_bets)
            .index
        )
        dropped = work.index[candidate].difference(kept)
        work.loc[dropped, 'skip_reason'] = REASON_MAX_BETS

    # [7] 賭け金は fractional Kelly（バックテストと同じ関数を使う）
    work['bet_amount'] = 0.0
    betting = work['skip_reason'] == REASON_BET
    if betting.any():
        work.loc[betting, 'bet_amount'] = evaluator._kelly_criterion_vec(
            work.loc[betting, 'pred_proba'].to_numpy(dtype='float64'),
            work.loc[betting, 'odds_used'].to_numpy(dtype='float64'),
            evaluator.initial_bankroll,
        )
        # Kelly が 0 円になった馬（オッズ対比で妙味が消えた）は買わない
        zero = betting & (work['bet_amount'] <= 0.0)
        work.loc[zero, 'skip_reason'] = REASON_KELLY

    return work.sort_values(
        ['race_id', 'pred_proba'], ascending=[True, False]
    ).reset_index(drop=True)


def _write_csv(
    frame: pd.DataFrame, path: Path, race_ids: set, *, merge_existing: bool
) -> pd.DataFrame:
    """CSV を書き出す。``merge_existing`` なら今回のレース以外の既存行を残す。

    Args:
        frame: 書き出す行
        path: 出力先
        race_ids: 今回予測したレース（既存行から置き換える対象）
        merge_existing: True なら既存ファイルのうち ``race_ids`` 以外の行を残す

    Returns:
        pd.DataFrame: 実際に書き出した表
    """
    if merge_existing and path.exists():
        existing = pd.read_csv(path, encoding='utf-8-sig', dtype={'race_id': str})
        existing = existing[~existing['race_id'].isin(race_ids)]
        frame = pd.concat([existing, frame], ignore_index=True, sort=False)
        frame = frame.sort_values(
            ['race_id', 'pred_proba'], ascending=[True, False], kind='stable'
        )
    frame.to_csv(path, index=False, encoding='utf-8-sig')
    return frame


def save_outputs(
    work: pd.DataFrame,
    target_date: dt.date,
    output_dir: Path,
    *,
    merge_existing: bool = False,
) -> Tuple[Path, Path]:
    """推奨 CSV と全馬 CSV を書き出す。

    Args:
        work: :func:`select_bets` の戻り値
        target_date: 対象日
        output_dir: 出力先ディレクトリ
        merge_existing: True なら既存の当日 CSV のうち今回予測していない
            レースの行を残して追記する（レース単位で予測する場合に使う）

    Returns:
        Tuple[Path, Path]: (推奨 CSV, 全馬 CSV) のパス
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f'{target_date:%Y%m%d}'
    race_ids = set(work['race_id'].astype(str))

    cols = [
        'race_id', 'horse_number', 'pred_proba', ODDS_COL, 'ev',
        'bet_amount', POOL_COL, INC_SHARE_COL, 'skip_reason',
    ]
    all_path = output_dir / f'{stamp}_all.csv'
    written_all = _write_csv(
        work[cols], all_path, race_ids, merge_existing=merge_existing
    )

    bets = work[work['skip_reason'] == REASON_BET]
    bets_path = output_dir / f'{stamp}_bets.csv'
    # 今回のレースに推奨が無い場合も、同レースの古い推奨行は置き換えて消す
    _write_csv(
        bets[cols].drop(columns='skip_reason'), bets_path, race_ids,
        merge_existing=merge_existing,
    )
    logger.warning('推奨を保存: %s（今回 %s 点）', bets_path, f'{len(bets):,}')
    logger.warning(
        '全馬の予測を保存: %s（今回 %s 行 / ファイル全体 %s 行）',
        all_path, f'{len(work):,}', f'{len(written_all):,}',
    )
    return bets_path, all_path


def run_predict_today(
    config: dict,
    *,
    target_date: Optional[dt.date] = None,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    min_ev: Optional[float] = None,
    fetch: bool = False,
    headless: bool = True,
    output_dir: Optional[Path] = None,
    race_ids: Optional[Iterable[str]] = None,
) -> Tuple[PredictionSummary, pd.DataFrame]:
    """当日予測を実行し、推奨 CSV を出力する。

    Args:
        config: 全体設定辞書
        target_date: 対象日（省略時は当日）
        realtime_dir: 当日オッズ CSV のディレクトリ
        model_filename: ``data/model`` 配下のモデルファイル名
        min_ev: EV しきい値（省略時はモデルの運用パラメータ）
        fetch: True なら当日スクレイパを起動してから予測する
        headless: 出馬表取得用ブラウザを非表示にするか
        output_dir: 出力先（省略時は ``config`` の ``output_dir/simulator``）
        race_ids: 指定したレースだけを予測し、当日 CSV へ追記する
            （省略時は取得済みの全レースを予測して上書きする）

    Returns:
        Tuple[PredictionSummary, pd.DataFrame]: (サマリ, 全馬の予測)
    """
    target_date = target_date or dt.date.today()

    if fetch:
        fetch_today_odds(
            target_date, realtime_dir=realtime_dir, headless=headless
        )

    # [1] 取得済み CSV
    raw = load_today_snapshots(target_date, realtime_dir, race_ids=race_ids)
    return predict_from_snapshots(
        config,
        raw,
        target_date=target_date,
        model_filename=model_filename,
        min_ev=min_ev,
        output_dir=output_dir,
        merge_existing=race_ids is not None,
    )


def predict_from_snapshots(
    config: dict,
    raw: pd.DataFrame,
    *,
    target_date: dt.date,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    min_ev: Optional[float] = None,
    output_dir: Optional[Path] = None,
    merge_existing: bool = False,
) -> Tuple[PredictionSummary, pd.DataFrame]:
    """読込済みの 10m / 5m ワイド表から予測し、推奨 CSV を出力する。

    Args:
        config: 全体設定辞書
        raw: :func:`src.simulator.realtime_loader.load_today_snapshots` の戻り値
        target_date: 対象日
        model_filename: ``data/model`` 配下のモデルファイル名
        min_ev: EV しきい値（省略時はモデルの運用パラメータ）
        output_dir: 出力先（省略時は ``config`` の ``output_dir/simulator``）
        merge_existing: True なら当日 CSV の他レースの行を残して追記する

    Returns:
        Tuple[PredictionSummary, pd.DataFrame]: (サマリ, 全馬の予測)
    """
    # [2][3] 特徴量
    work = attach_features(raw)
    n_races_total = int(work['race_id'].nunique())
    work = drop_incomplete_rows(work, context='当日予測')

    # [4] 予測
    proba, params = predict_proba(config, work, model_filename)

    # [5][6][7] フィルタ・EV・Kelly
    work = select_bets(config, work, proba, params, min_ev=min_ev)

    if output_dir is None:
        output_dir = Path(
            config.get('data', {}).get('output_dir', 'output')
        ) / OUTPUT_SUBDIR
    bets_path, all_path = save_outputs(
        work, target_date, output_dir, merge_existing=merge_existing
    )

    bets = work[work['skip_reason'] == REASON_BET]
    passed = work[work[POOL_COL] >= params.pool_threshold]
    summary = PredictionSummary(
        target_date=str(target_date),
        n_races_total=n_races_total,
        n_races_passed=int(passed['race_id'].nunique()) if len(passed) else 0,
        n_horses=len(work),
        n_bets=len(bets),
        total_bet=float(bets['bet_amount'].sum()),
        mean_odds=float(bets['odds_used'].mean()) if len(bets) else float('nan'),
        pool_threshold=params.pool_threshold,
        min_ev=float(min_ev if min_ev is not None else params.min_ev),
        bets_path=str(bets_path),
        all_path=str(all_path),
    )
    return summary, work


def _parse_date(value: Optional[str]) -> Optional[dt.date]:
    """``YYYY-MM-DD`` / ``YYYYMMDD`` を日付に変換する。

    Args:
        value: 日付文字列（None ならそのまま None を返す）

    Returns:
        Optional[dt.date]: 変換した日付

    Raises:
        ValueError: 形式が不正な場合
    """
    if value is None:
        return None
    for fmt in ('%Y-%m-%d', '%Y%m%d'):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f'日付は YYYY-MM-DD 形式で指定してください: {value}')


def _parse_args() -> argparse.Namespace:
    """コマンドライン引数を解析する。

    Returns:
        argparse.Namespace: 解析済み引数
    """
    parser = argparse.ArgumentParser(
        description='当日のオッズ・投票数で予測しベット推奨を出す（spec §6）'
    )
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='設定ファイルパス')
    parser.add_argument('--date', type=str, default=None,
                        help='対象日 (YYYY-MM-DD)。省略時は当日')
    parser.add_argument('--fetch', action='store_true',
                        help='当日スクレイパを起動してオッズを取得してから予測する')
    parser.add_argument('--headed', action='store_true',
                        help='--fetch 時に出馬表取得用ブラウザを表示する')
    parser.add_argument('--realtime-dir', type=str,
                        default=str(DEFAULT_REALTIME_DIR),
                        help='当日オッズ CSV のディレクトリ')
    parser.add_argument('--model-filename', type=str,
                        default=DEFAULT_MODEL_FILENAME,
                        help='data/model 配下のモデルファイル名')
    parser.add_argument('--min-ev', type=float, default=None,
                        help='EV しきい値（省略時はモデルの運用パラメータ）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力先（省略時は output/simulator）')
    return parser.parse_args()


def main() -> None:
    """CLI エントリポイント。"""
    from src.cli_common import build_config

    logging.basicConfig(
        level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s'
    )
    args = _parse_args()
    config = build_config(args.config)

    summary, work = run_predict_today(
        config,
        target_date=_parse_date(args.date),
        realtime_dir=Path(args.realtime_dir),
        model_filename=args.model_filename,
        min_ev=args.min_ev,
        fetch=args.fetch,
        headless=not args.headed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    bets = work[work['skip_reason'] == REASON_BET]
    print()
    print(f'=== 当日予測: {summary.target_date} ===')
    print(f'  レース      : {summary.n_races_passed} / {summary.n_races_total}'
          f' がフィルタ通過（{POOL_COL} >= {summary.pool_threshold:,.0f}円）')
    print(f'  賭け条件    : EV >= {summary.min_ev:.2f}')
    print(f'  推奨点数    : {summary.n_bets} 点 / 合計 {summary.total_bet:,.0f} 円')
    if summary.n_bets:
        print(f'  平均オッズ  : {summary.mean_odds:.1f} 倍')
        print()
        show = ['race_id', 'horse_number', 'pred_proba', ODDS_COL, 'ev', 'bet_amount']
        print(bets[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    else:
        print('  → 本日の推奨はありません（フィルタ・EV を通る馬がいませんでした）')
    print()
    print(f'  推奨CSV     : {summary.bets_path}')
    print(f'  全馬CSV     : {summary.all_path}')


if __name__ == '__main__':
    main()
