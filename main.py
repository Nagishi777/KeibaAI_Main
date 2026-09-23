"""KeibaAI 日次実行プログラム。レース当日の朝（8:30 目安）に起動する。

    [1] JRA出馬表から当日の発走時刻を取得
    [2] 各レースの 60m/30m/10m/5m/4m/3m/2m/1m/10s 前に JV-Link からオッズ・票数を取得
    [3] 各レースの 5m 前のオッズが揃ったら予測 → 購入ロジックに従って馬券を購入
    [4] 全レース終了後（最終レース発走 + 30分）にモデルを再学習

使い方::

    python main.py                  # src/buying/.env の BUYING_MODE に従って購入
    python main.py --paper          # ブラウザを開かず購入計画の検証だけ行う
    python main.py --no-buy         # 予測まで（購入処理は呼ばない）
    python main.py --live           # BUYING_MODE=live のときの実購入（明示が必須）
    python main.py --retrain-only   # 再学習だけすぐに実行

詳細は README.md を参照。
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='レース当日のオッズ取得・予測・購入・再学習を一括実行する',
    )
    parser.add_argument('--config', type=Path, default=Path('config/config.yaml'),
                        help='設定ファイル（既定: config/config.yaml）')
    parser.add_argument('--env', type=Path, default=Path('src/buying/.env'),
                        help='購入設定ファイル（既定: src/buying/.env）')

    buying = parser.add_mutually_exclusive_group()
    buying.add_argument('--no-buy', action='store_true',
                        help='購入処理を行わない（予測 CSV の出力まで）')
    buying.add_argument('--paper', action='store_true',
                        help='.env の設定に関わらず paper モード（ブラウザを開かない）で購入判定する')
    buying.add_argument('--live', action='store_true',
                        help='BUYING_MODE=live のときに実購入を許可する')

    retrain = parser.add_mutually_exclusive_group()
    retrain.add_argument('--no-retrain', action='store_true',
                         help='全レース終了後の再学習を行わない')
    retrain.add_argument('--retrain-only', action='store_true',
                         help='取得・予測・購入を行わず、再学習だけをすぐに実行する')

    parser.add_argument('--retrain-start-period', default=None,
                        help='再学習の開始年月 YYYYMM（既定: config の pipeline.retrain_start_period）')
    parser.add_argument('--retrain-delay-minutes', type=float, default=None,
                        help='最終レース発走から再学習開始までの分数（既定: config の値、無ければ 30）')
    parser.add_argument('--model-filename', default=None,
                        help='data/model 配下のモデルファイル名（既定: simulator/win_pool_filter.txt）')
    parser.add_argument('--min-ev', type=float, default=None,
                        help='EV しきい値（既定: モデルの運用パラメータ）')
    parser.add_argument('--include-past', action='store_true',
                        help='起動が遅れて過ぎた取得時点を JV-Link の時系列オッズから復元する')
    parser.add_argument('--headed', action='store_true',
                        help='出馬表取得用ブラウザを表示する')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # タスクスケジューラ等から起動されても相対パスが崩れないようにする
    os.chdir(PROJECT_ROOT)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    args = _parse_args(argv)

    from src.buying.config import ConfigurationError
    from src.buying.process_lock import AlreadyRunningError
    from src.pipeline.daily import (
        EXIT_ERROR,
        DailyPipeline,
        PipelineOptions,
        setup_pipeline_logging,
        today_jst,
    )

    target_date = today_jst()
    log_path = setup_pipeline_logging(target_date)
    logger = logging.getLogger('main')
    logger.warning('KeibaAI 日次実行を開始します: %s（ログ: %s）', target_date, log_path)

    options = PipelineOptions(
        config_path=args.config,
        env_path=args.env,
        model_filename=args.model_filename,
        min_ev=args.min_ev,
        buy=not args.no_buy,
        paper=args.paper,
        live=args.live,
        retrain=not args.no_retrain,
        retrain_only=args.retrain_only,
        retrain_start_period=args.retrain_start_period,
        retrain_delay_minutes=args.retrain_delay_minutes,
        headless=not args.headed,
        include_past=args.include_past,
    )
    try:
        return DailyPipeline(options, target_date=target_date).run()
    except (ConfigurationError, FileNotFoundError, AlreadyRunningError) as exc:
        logger.error('起動できませんでした: %s', exc)
        return EXIT_ERROR


if __name__ == '__main__':
    sys.exit(main())
