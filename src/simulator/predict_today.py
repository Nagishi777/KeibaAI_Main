"""当日のオッズ・投票数で予測しベット推奨を出す単独実行プログラム。

レース当日、発走2分前のオッズと総票数を取得して特徴量化し、学習済みモデルで
各馬の勝率を予測して、買うべき馬を CSV に出力する。

    # 取得から予測まで一気通貫（当日これを実行する）
    python -m src.simulator.predict_today --fetch

    # 既に取得済みの CSV から予測だけ行う
    python -m src.simulator.predict_today

    # 日付・EVしきい値を指定する
    python -m src.simulator.predict_today --date 2026-07-18 --min-ev 1.10

処理の流れ::

    [1] 発走2分前に、各馬の 2m オッズと総票数を取得
        （文脈として 60m/30m/10m/5m も読む）
    [2] 22列の特徴量を作り、LightGBM で各馬の勝率を予測（isotonic で校正）
    [3] 次の条件を全て満たす馬を買う
          ・EV = 予測確率 × 2mオッズ >= 1.05
          ・2mオッズが 10倍以上 150倍以下
          ・log(2mオッズ / 5mオッズ) <= -0.05   ← オッズが約5%以上下がった
          ・レースの2m時点総投票額が「その日の上位50%」
            （絶対額5億円は当日データのスケールでは使えない。下記 NOTE 参照）
    [4] 条件を満たす馬が複数いるレースでは期待値が最も高い1頭のみ
    [5] 賭け金は Flat ベット（1点 100円）

出力: ``output/simulator/{YYYYMMDD}_bets.csv``（推奨のみ）と
``{YYYYMMDD}_all.csv``（全馬の予測。どのフィルタで外れたかつき）。

NOTE (総投票額フィルタは分位点で行う):
    学習データ（JV-Link 履歴 0B41）と当日データ（速報 0B31）で
    ``hyosu_total`` のスケールが 9〜13倍違うため、絶対額（5億円）を
    当日データに当てると1レースも通らない。よって既定では
    「その日・その開催の中での分位点」で判定する
    （``config.yaml`` の ``strategy.min_pool_pctile``）。
    履歴データでの検証では、日内上位30%で回収率202.6%、
    絶対額7億円の200.7%とほぼ同等だった。
    詳細は ``docs/report/pool_by_race_number_report.md`` §6。

IMPORTANT (2m 時点で判断する理由):
    実際の投票締切は発走の約1分前であり、1mオッズを見てから発注する時間的
    余裕は無い。よって判断を 2m 時点に前倒ししている。
    2m以降のオッズ（1m・10s）は取得・保存するが予測には使わない。

IMPORTANT (学習は 1m / 当日判断は 2m):
    過去データに 2m スナップショットが存在しないため、モデルは 1m 時点で
    学習している。:mod:`src.simulator.features` の冒頭に記録した既知の
    制約であり、起動時にも警告が出る。

IMPORTANT (本条件の回収率は未検証の組み合わせである):
    レポートの検証値（回収率136.98% / 95%CI [109.3, 161.4]）は
    EV>=1.20・1レース最大3点・1m判断・fractional Kelly の構成である。
    本実装は EV>=1.05・1レース1点・2m判断・Flat100円に総投票額フィルタを
    加えた構成で、**この組み合わせ自体は検証されていない**。
    的中率5%・平均40倍の高分散戦略であることも併せて認識して運用すること。
"""
import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.model_creator import ModelCreator
from src.simulator.artifacts import (
    DEFAULT_MODEL_FILENAME,
    SimulatorParams,
    load_params,
    require_model,
    resolve_model_path,
)
from src.simulator.features import (
    DRIFT_COL,
    LIVE_DECISION_SNAPSHOT,
    POOL_COL,
    build_features,
    drop_incomplete_rows,
    feature_frame,
    normalize_late_columns,
)
from src.simulator.realtime_loader import (
    DEFAULT_REALTIME_DIR,
    load_today_snapshots,
)
from src.simulator.strategy import (
    REASON_BET,
    WinRule,
    apply_rule,
    rule_from_config,
)

logger = logging.getLogger(__name__)

BET_TYPE = 'win'

# 推奨 CSV の出力先（``output`` 配下）
OUTPUT_SUBDIR = 'simulator'

# 全馬 CSV に出す列（根拠が後から追えるようにする）
OUTPUT_COLS: Tuple[str, ...] = (
    'race_id', 'horse_number',
    'pred_proba', 'odds_used', 'ev',
    # 根拠となる市場の値
    'odds_5m_raw', DRIFT_COL, 'inc_share_5_late', POOL_COL,
    'mkt_rank_late', 'field_size',
    # どの時点を判断に使ったか（2m が欠けて代用した場合に効く）
    'decision_snapshot_used',
    # 見送り理由
    'skip_reason',
)


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
    rule_summary: str
    min_ev: float
    pool_filter_summary: str
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
    from src.scraping.realtime_odds import get_today_schedule, run_scheduler

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
        Tuple[np.ndarray, SimulatorParams]: (校正済み予測確率, 運用パラメータ)
    """
    creator = ModelCreator(config)
    model_path = resolve_model_path(creator.model_dir, model_filename)
    require_model(model_path)
    params = load_params(model_path)

    creator.load_model(model_filename, model_name=BET_TYPE)
    proba = creator.predict(feature_frame(df), model_name=BET_TYPE)
    return proba, params


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
    if 'ev' in frame.columns:
        frame = frame.sort_values(
            ['race_id', 'ev'], ascending=[True, False], kind='stable'
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

    全馬 CSV は「どのフィルタで外れたか」を ``skip_reason`` に残す。
    当日の意思決定では「なぜ買わなかったか」を後から確認できる必要がある。

    Args:
        work: :func:`src.simulator.strategy.apply_rule` の戻り値
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

    cols = [c for c in OUTPUT_COLS if c in work.columns]
    all_path = output_dir / f'{stamp}_all.csv'
    written_all = _write_csv(
        work[cols], all_path, race_ids, merge_existing=merge_existing
    )

    bets = work[work['skip_reason'] == REASON_BET]
    bets_path = output_dir / f'{stamp}_bets.csv'
    # 今回のレースに推奨が無い場合も、同レースの古い推奨行は置き換えて消す
    _write_csv(
        bets[[c for c in cols if c != 'skip_reason']], bets_path, race_ids,
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
        min_ev: EV しきい値（省略時は ``config`` / 既定値）
        fetch: True なら当日スクレイパを起動してから予測する
        headless: 出馬表取得用ブラウザを非表示にするか
        output_dir: 出力先（省略時は ``config`` の ``output_dir/simulator``）
        race_ids: 指定したレースだけを予測し、当日 CSV へ追記する

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
    """読込済みのワイド表から予測し、推奨 CSV を出力する。

    Args:
        config: 全体設定辞書
        raw: :func:`src.simulator.realtime_loader.load_today_snapshots` の戻り値
        target_date: 対象日
        model_filename: ``data/model`` 配下のモデルファイル名
        min_ev: EV しきい値（省略時は ``config`` / 既定値）
        output_dir: 出力先（省略時は ``config`` の ``output_dir/simulator``）
        merge_existing: True なら当日 CSV の他レースの行を残して追記する

    Returns:
        Tuple[PredictionSummary, pd.DataFrame]: (サマリ, 全馬の予測)
    """
    # [2] 特徴量。当日の「締切直前」は 2m。
    work = normalize_late_columns(raw, LIVE_DECISION_SNAPSHOT)
    work = build_features(work)
    n_races_total = int(work['race_id'].nunique())
    work = drop_incomplete_rows(work, context='当日予測')

    # [3] 予測（isotonic 校正済み）
    proba, _params = predict_proba(config, work, model_filename)

    # [4][5] 購入条件の適用と Flat ベット
    rule, _cfg = rule_from_config(config)
    if min_ev is not None:
        rule = WinRule(
            min_ev=float(min_ev),
            odds_min=rule.odds_min,
            odds_max=rule.odds_max,
            max_drift=rule.max_drift,
            min_pool_yen=rule.min_pool_yen,
            min_pool_pctile=rule.min_pool_pctile,
            max_bets_per_race=rule.max_bets_per_race,
            stake_yen=rule.stake_yen,
        )
    work = apply_rule(work, proba, rule)

    if output_dir is None:
        output_dir = Path(
            config.get('data', {}).get('output_dir', 'output')
        ) / OUTPUT_SUBDIR
    bets_path, all_path = save_outputs(
        work, target_date, output_dir, merge_existing=merge_existing
    )

    bets = work[work['skip_reason'] == REASON_BET]
    # 総投票額フィルタを通ったレース数（絶対額か分位点か、実際に使った方で数える）
    if rule.min_pool_pctile is not None:
        passed = work[work['pool_pctile'] >= rule.min_pool_pctile]
    else:
        passed = work[work[POOL_COL] >= rule.min_pool_yen]
    summary = PredictionSummary(
        target_date=str(target_date),
        n_races_total=n_races_total,
        n_races_passed=int(passed['race_id'].nunique()) if len(passed) else 0,
        n_horses=len(work),
        n_bets=len(bets),
        total_bet=float(len(bets) * rule.stake_yen),
        mean_odds=float(bets['odds_used'].mean()) if len(bets) else float('nan'),
        rule_summary=rule.describe(),
        min_ev=rule.min_ev,
        pool_filter_summary=(
            f'日内上位 {(1 - rule.min_pool_pctile) * 100:.0f}%'
            if rule.min_pool_pctile is not None
            else f'{rule.min_pool_yen / 1e8:.1f}億円 以上'
        ),
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
        description='当日のオッズ・投票数で予測しベット推奨を出す'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
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
                        help='EV しきい値（省略時は config / 既定 1.05）')
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
    print(f'  購入条件    : {summary.rule_summary}')
    print(f'  レース      : {summary.n_races_passed} / {summary.n_races_total}'
          f' が総投票額 {summary.pool_filter_summary}')
    print(f'  推奨点数    : {summary.n_bets} 点 / 合計 {summary.total_bet:,.0f} 円')
    if summary.n_bets:
        print(f'  平均オッズ  : {summary.mean_odds:.1f} 倍')
        print()
        show = ['race_id', 'horse_number', 'pred_proba', 'odds_used',
                DRIFT_COL, 'ev']
        print(bets[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    else:
        print('  → 本日の推奨はありません（条件を通る馬がいませんでした）')
    print()
    print(f'  推奨CSV     : {summary.bets_path}')
    print(f'  全馬CSV     : {summary.all_path}')


if __name__ == '__main__':
    main()
