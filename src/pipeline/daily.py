"""レース当日の処理を1プロセスで順に実行する日次パイプライン。

    [1] JRA出馬表から当日の発走時刻を取得する
        → data/processed/races/YYYYMMDD_jra_today_schedule.csv
    [2] 発走時刻に合わせて 60m〜10s の各時点で JV-Link からオッズ・票数を取得する
        → data/processed/realtime_odds/YYYYMMDD/{race_id}_{券種}_realtimeodds.csv
    [3] 各レースの 5m 時点が保存されたら、そのレースを予測し購入処理を行う
        → output/simulator/YYYYMMDD_{all,bets}.csv / src.buying の台帳・監査ログ
    [4] 全レース終了後（最終レース発走 + 待機時間）にモデルを再学習する
        → data/model/simulator/win_pool_filter*.{txt,pkl,json}

スレッド構成::

    メインスレッド : run_scheduler（JV-Link COM を扱う唯一のスレッド）
    判断スレッド   : 5m が保存されたレースを受け取り、予測 → 購入を行う

購入処理（ブラウザ操作）に時間がかかっても、判断スレッドで動くため
他レースのオッズ取得は止まらない。CSV の読み書きは
:data:`src.scraping.realtime_odds.CSV_LOCK` で排他する。

IMPORTANT (購入の安全装置は src.buying に従う):
    購入モード（paper / dry-run / live）・kill switch・金額上限・締切判定は
    ``src/buying/.env`` と :class:`src.buying.risk.RiskGate` がそのまま適用される。
    live 購入には ``BUYING_MODE=live`` に加えて ``--live`` の明示が必要。
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import queue
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from src.buying.browser.playwright_client import PlaywrightIpatClient
from src.buying.cli import build_audit_logger
from src.buying.clock import JST
from src.buying.config import BuyingSettings, ConfigurationError, load_settings
from src.buying.domain import BuyingMode
from src.buying.ledger import Ledger
from src.buying.process_lock import ProcessLock
from src.buying.service import PurchaseService
from src.buying.snapshot_reader import SnapshotReader
from src.buying.strategy import SimulatorStrategy, StrategyResult
from src.cli_common import load_config, load_features_config
from src.scraping.realtime_odds import (
    CSV_LOCK,
    get_today_schedule,
    run_scheduler,
)
from src.simulator.artifacts import (
    DEFAULT_MODEL_FILENAME,
    load_params,
    require_model,
    resolve_model_path,
)
from src.simulator.features import DECISION_SNAPSHOT
from src.simulator.predict_today import predict_from_snapshots
from src.simulator.realtime_loader import load_today_snapshots
from src.simulator.retrain import run_retrain

logger = logging.getLogger(__name__)

# 判断スレッドへ渡す取得イベントの状態（saved_stale も予測は記録し、購入可否は RiskGate が決める）
DECISION_STATUSES = frozenset({'saved', 'saved_stale'})

# config.yaml の ``pipeline`` セクションが無い場合の既定値
DEFAULT_RETRAIN_START_PERIOD = '202401'
DEFAULT_RETRAIN_DELAY_MINUTES = 30.0

# 日次レポート・ログの出力先
PIPELINE_OUTPUT_SUBDIR = 'pipeline'
DEFAULT_LOG_DIR = Path('logs/pipeline')

# 終了コード
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_PURCHASE_UNKNOWN = 2
EXIT_INTERRUPTED = 130


@dataclass
class PipelineOptions:
    """``main.py`` の引数に対応する実行オプション。"""

    config_path: Path = Path('config/config.yaml')
    env_path: Path = Path('src/buying/.env')
    model_filename: Optional[str] = None
    min_ev: Optional[float] = None
    buy: bool = True
    paper: bool = False
    live: bool = False
    retrain: bool = True
    retrain_only: bool = False
    retrain_start_period: Optional[str] = None
    retrain_delay_minutes: Optional[float] = None
    headless: bool = True
    include_past: bool = False


@dataclass
class PipelineReport:
    """1日分の実行記録。``output/pipeline/YYYYMMDD_pipeline_report.json`` に保存する。"""

    target_date: str
    started_at: str
    buying_mode: str = 'disabled'
    model_ready: bool = False
    model_error: str = ''
    n_races: int = 0
    first_post: str = ''
    last_post: str = ''
    scheduler_completed: Optional[bool] = None
    decisions: list[dict[str, Any]] = field(default_factory=list)
    retrain: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    finished_at: str = ''

    @property
    def purchase_unknown(self) -> int:
        return sum(int(item.get('purchase', {}).get('unknown', 0)) for item in self.decisions)


def _now_jst_naive() -> dt.datetime:
    """スケジューラと同じ基準（日本時間の naive datetime）で現在時刻を返す。"""
    return dt.datetime.now(JST).replace(tzinfo=None)


def today_jst() -> dt.date:
    """日本時間の今日の日付。"""
    return _now_jst_naive().date()


def load_pipeline_config(config_path: Path) -> dict:
    """``config.yaml`` と同じフォルダの ``features.json`` を読む（ロギングは変更しない）。"""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f'設定ファイルが見つかりません: {config_path}')
    config = load_config(config_path)
    config['features_def'] = load_features_config(config_path.parent / 'features.json')
    return config


def setup_pipeline_logging(target_date: dt.date, log_dir: Path = DEFAULT_LOG_DIR) -> Path:
    """画面と日別ログファイルの両方へ INFO 以上を出す。

    Returns:
        Path: 日別ログファイルのパス
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{target_date:%Y%m%d}_pipeline.log'
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding='utf-8'),
    ):
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        root.addHandler(handler)
    return log_path


def check_model(config: dict, model_filename: str) -> Optional[str]:
    """予測に使うモデル一式が揃っているか確認する。

    Returns:
        Optional[str]: 問題があればその内容、無ければ None
    """
    model_dir = Path(config.get('model', {}).get('model_dir', 'data/model'))
    model_path = resolve_model_path(model_dir, model_filename)
    try:
        require_model(model_path)
        load_params(model_path)
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)
    return None


def decision_race_ids(events: list[dict[str, object]]) -> list[str]:
    """取得イベントから、判断時点（5m）が保存されたレースを取り出す。"""
    race_ids: list[str] = []
    for event in events:
        if (
            event.get('snapshot_label') == DECISION_SNAPSHOT
            and event.get('status') in DECISION_STATUSES
        ):
            race_id = str(event.get('race_id'))
            if race_id not in race_ids:
                race_ids.append(race_id)
    return race_ids


class BuyingPurchaser:
    """:mod:`src.buying` の購入処理（RiskGate・台帳・ブラウザ操作）を呼び出す。"""

    def __init__(self, settings: BuyingSettings, *, allow_live: bool) -> None:
        self.settings = settings
        self.allow_live = allow_live
        self.reader = SnapshotReader(settings.schedule_dir, settings.odds_dir)
        self.ledger = Ledger(settings.ledger_path)
        self.audit = build_audit_logger(settings)

    def execute(self, result: StrategyResult, *, target_date: dt.date) -> dict[str, Any]:
        with CSV_LOCK:
            races = self.reader.load_races(target_date)
            evidence = self.reader.load_evidence(target_date)
        service = PurchaseService(
            self.settings,
            self.ledger,
            self.audit,
            lambda: PlaywrightIpatClient(
                login_url=self.settings.login_url,
                headless=self.settings.headless,
            ),
        )
        report = service.execute(
            result,
            target_date=target_date,
            races=races,
            evidence=evidence,
            allow_live=self.allow_live,
        )
        return dict(report.__dict__)


class DecisionEngine:
    """5m が保存されたレースを予測し、推奨を購入処理へ渡す。"""

    def __init__(
        self,
        config: dict,
        *,
        target_date: dt.date,
        realtime_dir: Path,
        model_filename: str,
        min_ev: Optional[float] = None,
        output_dir: Optional[Path] = None,
        purchaser: Optional[BuyingPurchaser] = None,
    ) -> None:
        self.config = config
        self.target_date = target_date
        self.realtime_dir = Path(realtime_dir)
        self.model_filename = model_filename
        self.min_ev = min_ev
        self.output_dir = output_dir
        self.purchaser = purchaser
        self.strategy = SimulatorStrategy(bet_unit_yen=100)

    def decide(self, race_ids: list[str]) -> dict[str, Any]:
        """指定レースを予測し、購入処理まで行って結果を返す。"""
        record: dict[str, Any] = {
            'race_ids': list(race_ids),
            'started_at': _now_jst_naive().isoformat(timespec='seconds'),
        }
        with CSV_LOCK:
            raw = load_today_snapshots(
                self.target_date, self.realtime_dir, race_ids=race_ids
            )
        summary, frame = predict_from_snapshots(
            self.config,
            raw,
            target_date=self.target_date,
            model_filename=self.model_filename,
            min_ev=self.min_ev,
            output_dir=self.output_dir,
            merge_existing=True,
        )
        record['prediction'] = {
            'n_races': summary.n_races_total,
            'n_races_passed_pool': summary.n_races_passed,
            'n_horses': summary.n_horses,
            'n_bets': summary.n_bets,
            'total_bet_yen': summary.total_bet,
        }
        result = self.strategy.from_frame(
            frame, target_date=self.target_date, summary=summary
        )
        record['intents'] = [
            {
                'race_id': intent.race_id,
                'horse_number': intent.selection[0],
                'amount_yen': intent.amount_yen,
                'expected_odds': intent.expected_odds,
            }
            for intent in result.intents
        ]
        if self.purchaser is not None and result.intents:
            record['purchase'] = self.purchaser.execute(result, target_date=self.target_date)
        record['finished_at'] = _now_jst_naive().isoformat(timespec='seconds')
        logger.warning(
            '判断完了 %s: 推奨 %d 点 / 購入指示 %d 件 / 購入結果 %s',
            ','.join(race_ids), summary.n_bets, len(result.intents),
            record.get('purchase', '購入なし'),
        )
        return record


class RaceDecisionWorker(threading.Thread):
    """判断待ちのレースを順に処理する判断スレッド。

    処理中に届いたレースはまとめて1回で処理する（ブラウザのログインを1回に抑える）。
    """

    _STOP = object()

    def __init__(
        self,
        decide: Callable[[list[str]], dict[str, Any]],
        *,
        on_result: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        super().__init__(name='race-decision', daemon=True)
        self._decide = decide
        self._on_result = on_result
        self._queue: queue.Queue = queue.Queue()
        self._submitted: set[str] = set()
        self.results: list[dict[str, Any]] = []

    def submit(self, race_ids: list[str]) -> None:
        """レースを判断待ちに追加する（同じレースは1日1回だけ処理する）。"""
        for race_id in race_ids:
            if race_id in self._submitted:
                continue
            self._submitted.add(race_id)
            self._queue.put(race_id)

    def stop(self) -> None:
        """待ち行列を処理し終えたら終了させ、終了まで待つ。"""
        self._queue.put(self._STOP)
        self.join()

    def run(self) -> None:
        stopping = False
        while not stopping:
            item = self._queue.get()
            if item is self._STOP:
                break
            batch = [item]
            while True:
                try:
                    extra = self._queue.get_nowait()
                except queue.Empty:
                    break
                if extra is self._STOP:
                    stopping = True
                    break
                batch.append(extra)
            self._handle(batch)

    def _handle(self, batch: list[str]) -> None:
        try:
            record = self._decide(batch)
        except Exception as exc:
            logger.exception('レース %s の予測・購入でエラーが発生しました', ','.join(batch))
            record = {
                'race_ids': batch,
                'error': f'{type(exc).__name__}: {exc}',
            }
        self.results.append(record)
        if self._on_result is not None:
            try:
                self._on_result(record)
            except Exception:
                logger.exception('判断結果の記録に失敗しました')


def sleep_until(
    target: dt.datetime,
    *,
    now: Callable[[], dt.datetime] = _now_jst_naive,
    sleep: Callable[[float], None] = time.sleep,
    max_step_seconds: float = 60.0,
) -> None:
    """``target``（日本時間 naive）まで待つ。"""
    while True:
        remaining = (target - now()).total_seconds()
        if remaining <= 0:
            return
        sleep(min(remaining, max_step_seconds))


def run_retrain_step(
    config: dict,
    *,
    target_date: dt.date,
    start_period: str,
    model_filename: str,
) -> dict[str, Any]:
    """学習開始月〜対象日の月でモデルを再学習する。"""
    end_period = f'{target_date:%Y%m}'
    logger.warning('再学習を開始します: %s 〜 %s', start_period, end_period)
    result = run_retrain(
        config,
        start_period=start_period,
        end_period=end_period,
        model_filename=model_filename,
    )
    logger.warning(
        '再学習完了: %s（学習期間 %s〜%s / %s レース / holdout AUC %.4f / しきい値 %s円）',
        result.model_path, result.train_start, result.train_end,
        f'{result.n_races:,}', result.holdout_auc, f'{result.pool_threshold:,.0f}',
    )
    if result.train_end < target_date.isoformat():
        logger.warning(
            '学習データの最終日は %s です。当日（%s）のレース結果は学習に含まれていません。'
            ' data/processed/odds_series と data/processed/horses を更新してから'
            ' python main.py --retrain-only を実行すると、当日分を含めて再学習できます。',
            result.train_end, target_date,
        )
    return asdict(result)


class DailyPipeline:
    """[1]〜[4] を順に実行する。"""

    def __init__(
        self,
        options: PipelineOptions,
        *,
        target_date: Optional[dt.date] = None,
        output_root: Optional[Path] = None,
    ) -> None:
        self.options = options
        self.target_date = target_date or _now_jst_naive().date()
        self.config = load_pipeline_config(options.config_path)
        pipeline_cfg = self.config.get('pipeline', {}) or {}
        self.model_filename = (
            options.model_filename
            or pipeline_cfg.get('model_filename')
            or DEFAULT_MODEL_FILENAME
        )
        self.retrain_start_period = str(
            options.retrain_start_period
            or pipeline_cfg.get('retrain_start_period')
            or DEFAULT_RETRAIN_START_PERIOD
        )
        self.retrain_delay = dt.timedelta(minutes=float(
            options.retrain_delay_minutes
            if options.retrain_delay_minutes is not None
            else pipeline_cfg.get('retrain_delay_minutes', DEFAULT_RETRAIN_DELAY_MINUTES)
        ))
        self.output_root = Path(
            output_root or self.config.get('data', {}).get('output_dir', 'output')
        )
        self.report = PipelineReport(
            target_date=self.target_date.isoformat(),
            started_at=_now_jst_naive().isoformat(timespec='seconds'),
        )
        self._report_lock = threading.Lock()

    # ------------------------------------------------------------------
    # 実行
    # ------------------------------------------------------------------

    def run(self) -> int:
        """パイプラインを実行し、終了コードを返す。"""
        try:
            if self.options.retrain_only:
                return self._finish(self._retrain(wait_until=None))
            return self._finish(self._run_race_day())
        except KeyboardInterrupt:
            logger.warning('Ctrl+C で停止しました')
            self.report.errors.append('interrupted')
            return self._finish(EXIT_INTERRUPTED)
        except ConfigurationError:
            raise
        except Exception as exc:
            logger.exception('日次実行を中断しました')
            self.report.errors.append(f'{type(exc).__name__}: {exc}')
            return self._finish(EXIT_ERROR)

    def _run_race_day(self) -> int:
        settings = self._buying_settings()
        schedule_dir = settings.schedule_dir
        realtime_dir = settings.odds_dir

        model_error = check_model(self.config, self.model_filename)
        self.report.model_ready = model_error is None
        if model_error is not None:
            self.report.model_error = model_error
            logger.error(
                'モデルが使えないため、本日は予測・購入を行わずオッズ取得と再学習のみ行います: %s',
                model_error,
            )

        with ExitStack() as stack:
            purchaser: Optional[BuyingPurchaser] = None
            if self.options.buy and model_error is None:
                stack.enter_context(
                    ProcessLock(settings.data_dir / f'{settings.account_alias}.lock')
                )
                purchaser = BuyingPurchaser(settings, allow_live=self.options.live)

            # [1] 当日の発走時刻
            try:
                schedule = get_today_schedule(
                    self.target_date,
                    headless=self.options.headless,
                    schedule_dir=schedule_dir,
                )
            except RuntimeError as exc:
                logger.error('本日の開催が無いか、出馬表を取得できませんでした: %s', exc)
                self.report.errors.append(f'schedule: {exc}')
                return EXIT_ERROR
            self._record_schedule(schedule)

            # [3] 判断スレッド
            worker: Optional[RaceDecisionWorker] = None
            if model_error is None:
                engine = DecisionEngine(
                    self.config,
                    target_date=self.target_date,
                    realtime_dir=realtime_dir,
                    model_filename=self.model_filename,
                    min_ev=self.options.min_ev,
                    output_dir=self.output_root / 'simulator',
                    purchaser=purchaser,
                )
                worker = RaceDecisionWorker(engine.decide, on_result=self._record_decision)
                worker.start()

            def on_jobs_executed(events: list[dict[str, object]]) -> None:
                race_ids = decision_race_ids(events)
                if race_ids and worker is not None:
                    logger.info('%s 時点を保存したレースを判断待ちに追加: %s',
                                DECISION_SNAPSHOT, ','.join(race_ids))
                    worker.submit(race_ids)

            # [2] 時点別オッズ（最終レースの 10s 前まで動き続ける）
            completed = run_scheduler(
                schedule,
                self.target_date,
                output_dir=realtime_dir,
                include_past=self.options.include_past,
                on_jobs_executed=on_jobs_executed,
            )
            self.report.scheduler_completed = completed
            if worker is not None:
                logger.info('判断待ちのレースを処理し終えるまで待機します')
                worker.stop()
            self._save_report()
            if not completed:
                logger.warning('オッズ取得を中断したため再学習は行いません')
                return EXIT_INTERRUPTED

        # [4] 再学習
        if not self.options.retrain:
            logger.warning('--no-retrain が指定されたため再学習を行いません')
            return self._exit_code()
        last_post = dt.datetime.fromisoformat(self.report.last_post)
        return self._retrain(wait_until=last_post + self.retrain_delay)

    def _retrain(self, *, wait_until: Optional[dt.datetime]) -> int:
        if wait_until is not None:
            logger.warning('再学習は %s に開始します（最終レース発走 + %s 分）',
                           wait_until.strftime('%H:%M'),
                           int(self.retrain_delay.total_seconds() // 60))
            sleep_until(wait_until)
        try:
            self.report.retrain = run_retrain_step(
                self.config,
                target_date=self.target_date,
                start_period=self.retrain_start_period,
                model_filename=self.model_filename,
            )
        except Exception as exc:
            logger.exception('再学習に失敗しました（既存のモデルはそのまま残ります）')
            self.report.retrain = {'error': f'{type(exc).__name__}: {exc}'}
            self.report.errors.append(f'retrain: {exc}')
            return EXIT_ERROR
        return self._exit_code()

    # ------------------------------------------------------------------
    # 補助
    # ------------------------------------------------------------------

    def _buying_settings(self) -> BuyingSettings:
        """購入設定を読み、実行モードを確定する（問題があれば起動時に止める）。"""
        options = self.options
        if not options.buy:
            settings = load_settings(options.env_path, require_credentials=False)
            self.report.buying_mode = 'disabled'
            logger.warning('--no-buy が指定されたため購入は行いません（予測のみ）')
            return settings
        if options.paper:
            settings = replace(
                load_settings(options.env_path, require_credentials=False),
                mode=BuyingMode.PAPER,
            )
        else:
            settings = load_settings(options.env_path)
        if settings.mode == BuyingMode.LIVE and not options.live:
            raise ConfigurationError(
                'BUYING_MODE=live の実購入には --live の指定が必要です'
                '（誤って実購入しないための確認です）'
            )
        if options.live and settings.mode != BuyingMode.LIVE:
            logger.warning('--live は BUYING_MODE=live のときだけ有効です（現在 %s）',
                           settings.mode.value)
        if settings.kill_switch:
            logger.warning(
                'BUYING_KILL_SWITCH=true のため購入は全て見送られます（判定と記録のみ）'
            )
        self.report.buying_mode = settings.mode.value
        logger.warning('購入モード: %s', settings.mode.value)
        return settings

    def _record_schedule(self, schedule: Any) -> None:
        posts = schedule['post_datetime']
        self.report.n_races = int(len(schedule))
        self.report.first_post = posts.min().isoformat()
        self.report.last_post = posts.max().isoformat()
        logger.warning(
            '本日のレース: %d レース（最初 %s / 最終 %s）',
            self.report.n_races,
            posts.min().strftime('%H:%M'),
            posts.max().strftime('%H:%M'),
        )
        self._save_report()

    def _record_decision(self, record: dict[str, Any]) -> None:
        with self._report_lock:
            self.report.decisions.append(record)
            races = ','.join(record.get('race_ids', []))
            if 'error' in record:
                self.report.errors.append(f"decision {races}: {record['error']}")
            for error in record.get('purchase', {}).get('errors', []):
                self.report.errors.append(f'purchase {races}: {error}')
        self._save_report()

    def _exit_code(self) -> int:
        if self.report.purchase_unknown:
            logger.error(
                '成立不明の購入が %d 件あります。python -m src.buying.cli reconcile で確認し、'
                '即PATの投票内容照会と照合してください（自動再送はしません）。',
                self.report.purchase_unknown,
            )
            return EXIT_PURCHASE_UNKNOWN
        return EXIT_ERROR if self.report.errors else EXIT_OK

    def report_path(self) -> Path:
        return (
            self.output_root / PIPELINE_OUTPUT_SUBDIR
            / f'{self.target_date:%Y%m%d}_pipeline_report.json'
        )

    def _save_report(self) -> Path:
        path = self.report_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._report_lock:
            payload = asdict(self.report)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding='utf-8',
        )
        return path

    def _finish(self, code: int) -> int:
        self.report.finished_at = _now_jst_naive().isoformat(timespec='seconds')
        path = self._save_report()
        logger.warning('日次レポート: %s（終了コード %d）', path, code)
        return code
