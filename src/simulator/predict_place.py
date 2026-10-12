"""当日のオッズで複勝率を予測し、複勝のベット推奨を出す単独実行プログラム。

    # 取得済みの当日 CSV から予測する
    python -m src.simulator.predict_place --date 2026-10-11

処理の流れ（``docs/report/anaba_strategy_place_report/anaba_place_band_report.md`` §4）::

    [1] 判断時点（2m）の単勝オッズ（文脈として 60m/30m/10m/5m）と総票数、
        および同じ時点の複勝オッズ（min/max）を読む
    [2] 単勝版と同じ22列の特徴量を作り、複勝専用モデルで複勝率を予測
        （5シード・各シードを isotonic で校正して平均）
    [3] 次の3条件を全て満たす馬を買う
          ・複勝EV = 予測複勝率 × 複勝オッズ(min側) >= 1.05
          ・複勝オッズ(min側)が 6倍以上 20倍以下
          ・log(単勝2m / 単勝5m) <= -0.05
    [4] 1レース最大3点（期待値の高い順）・Flat 100円

出力: ``output/simulator/{YYYYMMDD}_place_bets.csv``（推奨のみ）と
``{YYYYMMDD}_place_all.csv``（全馬。どのフィルタで外れたかつき）。

IMPORTANT (検証構成との差分):
    レポートの検証は **1m** 判断・確定払戻で決済した数字である
    （回収率 140.43% CV / 144.54% 真のOOS、95%CI 下側 107〜113%）。
    本実装は単勝版と同じく **2m** 判断であり、この時点での成績は検証されていない。
    計画は点推定ではなく CI 下側を基礎にすること（レポート §8）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import pandas as pd

from src.simulator.features import (
    DRIFT_COL,
    LIVE_DECISION_SNAPSHOT,
    build_features,
    drop_incomplete_rows,
    normalize_late_columns,
)
from src.simulator.place_model import PlaceModel, resolve_model_dir
from src.simulator.place_strategy import (
    PLACE_ODDS_COL,
    PLACE_ODDS_MAX_COL,
    apply_place_rule,
    place_rule_from_config,
)
from src.simulator.predict_today import OUTPUT_SUBDIR, _parse_date, _write_csv
from src.simulator.realtime_loader import (
    DEFAULT_REALTIME_DIR,
    load_today_place_odds,
    load_today_snapshots,
)
from src.simulator.strategy import REASON_BET

logger = logging.getLogger(__name__)

BET_TYPE = 'place'

# 出力ファイル名の接尾辞（単勝の {stamp}_all.csv と並べて置く）
FILE_TAG = 'place'

OUTPUT_COLS: Tuple[str, ...] = (
    'race_id', 'horse_number',
    'pred_proba', 'odds_used', PLACE_ODDS_MAX_COL, 'ev',
    'odds_late_raw', 'odds_5m_raw', DRIFT_COL, 'inc_share_5_late',
    'mkt_rank_late', 'field_size',
    'decision_snapshot_used',
    'skip_reason',
)


@dataclass
class PlacePredictionSummary:
    """当日複勝予測のサマリ。"""

    target_date: str
    n_races_total: int
    n_horses: int
    n_bets: int
    total_bet: float
    mean_odds: float
    rule_summary: str
    min_ev: float
    bets_path: str
    all_path: str


def save_place_outputs(
    work: pd.DataFrame,
    target_date: dt.date,
    output_dir: Path,
    *,
    merge_existing: bool = False,
) -> Tuple[Path, Path]:
    """推奨 CSV と全馬 CSV を書き出す（単勝版と同じ追記・置換の規則）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f'{target_date:%Y%m%d}'
    race_ids = set(work['race_id'].astype(str))
    cols = [c for c in OUTPUT_COLS if c in work.columns]

    all_path = output_dir / f'{stamp}_{FILE_TAG}_all.csv'
    written_all = _write_csv(work[cols], all_path, race_ids, merge_existing=merge_existing)
    bets = work[work['skip_reason'] == REASON_BET]
    bets_path = output_dir / f'{stamp}_{FILE_TAG}_bets.csv'
    _write_csv(
        bets[[c for c in cols if c != 'skip_reason']], bets_path, race_ids,
        merge_existing=merge_existing,
    )
    logger.warning('[複勝] 推奨を保存: %s（今回 %s 点）', bets_path, f'{len(bets):,}')
    logger.warning(
        '[複勝] 全馬の予測を保存: %s（今回 %s 行 / ファイル全体 %s 行）',
        all_path, f'{len(work):,}', f'{len(written_all):,}',
    )
    return bets_path, all_path


def attach_place_odds(
    work: pd.DataFrame,
    target_date: dt.date,
    realtime_dir: Path,
    place_odds: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """単勝の判断時点と同じ時点の複勝オッズを結合する（無い馬は NaN のまま）。

    ``place_odds`` を渡せばファイルを読まずにそれを使う（日次パイプラインは
    CSV の排他ロック内で先に読んでおく）。
    """
    if place_odds is None:
        snapshot_by_race = (
            work.groupby('race_id')['decision_snapshot_used'].first().astype(str).to_dict()
            if 'decision_snapshot_used' in work.columns
            else {race_id: LIVE_DECISION_SNAPSHOT for race_id in work['race_id'].unique()}
        )
        place_odds = load_today_place_odds(target_date, realtime_dir, snapshot_by_race)
    place = place_odds.copy()
    place['race_id'] = place['race_id'].astype(str)
    place['horse_number'] = place['horse_number'].astype('int64')
    out = work.drop(columns=[PLACE_ODDS_COL, PLACE_ODDS_MAX_COL], errors='ignore')
    out = out.merge(place, on=['race_id', 'horse_number'], how='left')
    n_missing = int(out[PLACE_ODDS_COL].isna().sum())
    if n_missing:
        logger.warning('[複勝] 複勝オッズが無い %s 頭は購入対象外にします', f'{n_missing:,}')
    return out


def predict_place_from_snapshots(
    config: dict,
    raw: pd.DataFrame,
    *,
    target_date: dt.date,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    model_subdir: Optional[str] = None,
    min_ev: Optional[float] = None,
    output_dir: Optional[Path] = None,
    merge_existing: bool = False,
    model: Optional[PlaceModel] = None,
    place_odds: Optional[pd.DataFrame] = None,
) -> Tuple[PlacePredictionSummary, pd.DataFrame]:
    """読込済みの単勝ワイド表から複勝を予測し、推奨 CSV を出力する。

    Args:
        config: 全体設定辞書
        raw: :func:`src.simulator.realtime_loader.load_today_snapshots` の戻り値
        target_date: 対象日
        realtime_dir: 当日オッズ CSV のディレクトリ（複勝 CSV を読む）
        model_subdir: ``data/model`` 配下のモデルフォルダ
        min_ev: EV しきい値（省略時は ``config`` / 既定 1.05）
        output_dir: 出力先（省略時は ``output/simulator``）
        merge_existing: True なら当日 CSV の他レースの行を残して追記する
        model: 読込済みモデル（省略時はここで読む）
        place_odds: 読込済みの複勝オッズ（省略時は ``realtime_dir`` から読む）

    Returns:
        Tuple[PlacePredictionSummary, pd.DataFrame]: (サマリ, 全馬の予測)
    """
    work = normalize_late_columns(raw, LIVE_DECISION_SNAPSHOT)
    work = build_features(work)
    n_races_total = int(work['race_id'].nunique())
    work = drop_incomplete_rows(work, context='当日予測（複勝）')
    work['race_id'] = work['race_id'].astype(str)
    work = attach_place_odds(work, target_date, realtime_dir, place_odds)

    model = model or PlaceModel.load(resolve_model_dir(config, model_subdir))
    proba = model.predict(work)

    rule, _cfg = place_rule_from_config(config, min_ev=min_ev)
    work = apply_place_rule(work, proba, rule)

    if output_dir is None:
        output_dir = Path(config.get('data', {}).get('output_dir', 'output')) / OUTPUT_SUBDIR
    bets_path, all_path = save_place_outputs(
        work, target_date, output_dir, merge_existing=merge_existing
    )
    bets = work[work['skip_reason'] == REASON_BET]
    summary = PlacePredictionSummary(
        target_date=str(target_date),
        n_races_total=n_races_total,
        n_horses=len(work),
        n_bets=len(bets),
        total_bet=float(len(bets) * rule.stake_yen),
        mean_odds=float(bets['odds_used'].mean()) if len(bets) else float('nan'),
        rule_summary=rule.describe(),
        min_ev=rule.min_ev,
        bets_path=str(bets_path),
        all_path=str(all_path),
    )
    return summary, work


def run_predict_place_today(
    config: dict,
    *,
    target_date: Optional[dt.date] = None,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    model_subdir: Optional[str] = None,
    min_ev: Optional[float] = None,
    output_dir: Optional[Path] = None,
    race_ids: Optional[Iterable[str]] = None,
) -> Tuple[PlacePredictionSummary, pd.DataFrame]:
    """取得済みの当日 CSV から複勝を予測する。"""
    target_date = target_date or dt.date.today()
    raw = load_today_snapshots(target_date, realtime_dir, race_ids=race_ids)
    return predict_place_from_snapshots(
        config, raw,
        target_date=target_date,
        realtime_dir=realtime_dir,
        model_subdir=model_subdir,
        min_ev=min_ev,
        output_dir=output_dir,
        merge_existing=race_ids is not None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='当日のオッズで複勝の推奨を出す')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--date', type=str, default=None, help='対象日 (YYYY-MM-DD)。省略時は当日')
    parser.add_argument('--realtime-dir', type=str, default=str(DEFAULT_REALTIME_DIR))
    parser.add_argument('--model-dir', type=str, default=None,
                        help='data/model 配下の複勝モデルフォルダ（既定 place）')
    parser.add_argument('--min-ev', type=float, default=None,
                        help='複勝EV しきい値（省略時は config / 既定 1.05）')
    parser.add_argument('--output-dir', type=str, default=None)
    return parser.parse_args()


def main() -> None:
    """CLI エントリポイント。"""
    from src.cli_common import build_config

    logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s] %(message)s')
    args = _parse_args()
    config = build_config(args.config)
    summary, work = run_predict_place_today(
        config,
        target_date=_parse_date(args.date),
        realtime_dir=Path(args.realtime_dir),
        model_subdir=args.model_dir,
        min_ev=args.min_ev,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    bets = work[work['skip_reason'] == REASON_BET]
    print()
    print(f'=== 当日予測（複勝）: {summary.target_date} ===')
    print(f'  購入条件    : {summary.rule_summary}')
    print(f'  レース      : {summary.n_races_total}')
    print(f'  推奨点数    : {summary.n_bets} 点 / 合計 {summary.total_bet:,.0f} 円')
    if summary.n_bets:
        print(f'  平均複勝オッズ(min): {summary.mean_odds:.1f} 倍')
        print()
        show = ['race_id', 'horse_number', 'pred_proba', 'odds_used', PLACE_ODDS_MAX_COL,
                DRIFT_COL, 'ev']
        print(bets[show].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
    else:
        print('  → 推奨はありません（条件を通る馬がいませんでした）')
    print()
    print(f'  推奨CSV     : {summary.bets_path}')
    print(f'  全馬CSV     : {summary.all_path}')


if __name__ == '__main__':
    main()
