"""即PAT購入システムのCLI。

例::

    python -m src.buying.cli doctor
    python -m src.buying.cli paper --date 2026-09-21 --bets-csv output/simulator/20260921_bets.csv
    python -m src.buying.cli dry-run --date 2026-09-21
    python -m src.buying.cli run-once --live
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.buying.audit import AuditLogger
from src.buying.browser.playwright_client import PlaywrightIpatClient
from src.buying.clock import JST, now_jst
from src.buying.config import BuyingSettings, ConfigurationError, load_settings
from src.buying.domain import BuyingMode
from src.buying.ledger import Ledger
from src.buying.process_lock import ProcessLock
from src.buying.service import PurchaseService
from src.buying.snapshot_reader import SnapshotReader
from src.buying.strategy import SimulatorStrategy, StrategyResult


def _date(value: str | None) -> dt.date:
    if value is None:
        return now_jst().date()
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"日付はYYYY-MM-DDまたはYYYYMMDDです: {value}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JRA即PAT 自動購入システム")
    parser.add_argument("--env", type=Path, default=Path("src/buying/.env"))
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="設定・依存関係・入力ファイルを診断")
    for name, help_text in (
        ("paper", "ブラウザを開かず購入計画だけ検証"),
        ("dry-run", "即PAT確認画面まで操作するが送信しない"),
        ("run-once", "現在の設定モードで1回実行"),
        ("run-day", "当日の開催終了まで定期実行"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        _add_run_arguments(cmd)
        if name in {"run-once", "run-day"}:
            cmd.add_argument(
                "--live",
                action="store_true",
                help="BUYING_MODE=liveの場合に実購入を明示許可",
            )
    sub.add_parser("reconcile", help="成立不明の台帳項目を表示（自動再送しない）")
    return parser


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--date", default=None, help="対象日 YYYY-MM-DD（省略時はJST当日）")
    parser.add_argument("--bets-csv", type=Path, default=None, help="生成済み *_bets.csv")
    parser.add_argument("--simulator-config", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--model-filename", default=None)
    parser.add_argument("--min-ev", type=float, default=None)
    parser.add_argument("--fetch", action="store_true", help="シミュレータの取得処理も呼ぶ")


def _settings_for_command(args: argparse.Namespace) -> BuyingSettings:
    if args.command == "paper":
        return replace(load_settings(args.env, require_credentials=False), mode=BuyingMode.PAPER)
    if args.command == "dry-run":
        return replace(load_settings(args.env, require_credentials=True), mode=BuyingMode.DRY_RUN)
    return load_settings(args.env, require_credentials=args.command not in {"doctor", "reconcile"})


def _strategy(args: argparse.Namespace, settings: BuyingSettings, target_date: dt.date) -> StrategyResult:
    strategy = SimulatorStrategy(bet_unit_yen=100)
    if args.bets_csv is not None:
        return strategy.from_csv(args.bets_csv, target_date=target_date)
    return strategy.run(
        target_date=target_date,
        simulator_config_path=args.simulator_config,
        realtime_dir=settings.odds_dir,
        model_filename=args.model_filename,
        min_ev=args.min_ev,
        fetch=args.fetch,
        headless=settings.headless,
    )


def _audit(settings: BuyingSettings) -> AuditLogger:
    credentials = settings.credentials
    secrets = () if credentials is None else (
        credentials.inet_id,
        credentials.subscriber_number,
        credentials.pin,
        credentials.pars_number,
    )
    return AuditLogger(settings.log_dir, secrets=secrets)


def _run_once(
    args: argparse.Namespace,
    settings: BuyingSettings,
    *,
    target_date: dt.date,
) -> dict:
    reader = SnapshotReader(settings.schedule_dir, settings.odds_dir)
    races = reader.load_races(target_date)
    evidence = reader.load_evidence(target_date)
    result = _strategy(args, settings, target_date)
    ledger = Ledger(settings.ledger_path)
    audit = _audit(settings)
    service = PurchaseService(
        settings,
        ledger,
        audit,
        lambda: PlaywrightIpatClient(
            login_url=settings.login_url,
            headless=settings.headless,
        ),
    )
    report = service.execute(
        result,
        target_date=target_date,
        races=races,
        evidence=evidence,
        allow_live=bool(getattr(args, "live", False)),
    )
    return report.__dict__


def _run_day(args: argparse.Namespace, settings: BuyingSettings, target_date: dt.date) -> int:
    reader = SnapshotReader(settings.schedule_dir, settings.odds_dir)
    races = reader.load_races(target_date)
    if not races:
        raise ValueError(f"対象日のレースがありません: {target_date}")
    stop_at = max(race.post_datetime for race in races.values()) + dt.timedelta(minutes=5)
    while now_jst() <= stop_at:
        try:
            report = _run_once(args, settings, target_date=target_date)
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        except (FileNotFoundError, ValueError) as exc:
            # 1mファイルがまだ生成されていない時間帯は次のポーリングで再試行する。
            print(f"待機: {exc}", file=sys.stderr)
        time.sleep(settings.poll_interval_seconds)
    return 0


def _doctor(settings: BuyingSettings, env_path: Path) -> int:
    checks: list[tuple[str, bool, str]] = []
    checks.append(("env", env_path.exists(), str(env_path)))
    for name, module in (
        ("pandas", "pandas"),
        ("playwright", "playwright"),
        ("yaml", "yaml"),
        ("lightgbm", "lightgbm"),
        ("sklearn", "sklearn"),
    ):
        try:
            importlib.import_module(module)
            checks.append((name, True, "import可"))
        except Exception as exc:
            checks.append((name, False, str(exc)))
    checks.append(("schedule_dir", settings.schedule_dir.exists(), str(settings.schedule_dir)))
    checks.append(("odds_dir", settings.odds_dir.exists(), str(settings.odds_dir)))
    checks.append(("credentials", settings.credentials is not None, "設定済み" if settings.credentials else "未設定"))
    checks.append(("kill_switch", not settings.kill_switch, "OFF" if not settings.kill_switch else "ON（購入停止）"))
    for name, ok, detail in checks:
        print(f"[{'OK' if ok else 'NG'}] {name}: {detail}")
    # kill switchと認証情報はpaper利用時には意図的であり得るので終了コードへ含めない。
    required = [ok for name, ok, _ in checks if name not in {"credentials", "kill_switch", "env"}]
    return 0 if all(required) else 1


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        settings = _settings_for_command(args)
        if args.command == "doctor":
            raise SystemExit(_doctor(settings, args.env))
        if args.command == "reconcile":
            entries = Ledger(settings.ledger_path).unknown_entries()
            if not entries:
                print("成立不明の購入はありません。")
                return
            print("自動再送は禁止です。即PATの投票内容照会で次を確認してください。")
            for entry in entries:
                print(
                    f"- state={entry.state} race_id={entry.race_id} "
                    f"amount={entry.amount_yen} key={entry.idempotency_key[:12]}"
                )
            return

        target_date = _date(args.date)
        lock_path = settings.data_dir / f"{settings.account_alias}.lock"
        with ProcessLock(lock_path):
            if args.command == "run-day":
                raise SystemExit(_run_day(args, settings, target_date))
            report = _run_once(args, settings, target_date=target_date)
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        if report["unknown"]:
            raise SystemExit(2)
        if report["errors"]:
            raise SystemExit(1)
    except (ConfigurationError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()

