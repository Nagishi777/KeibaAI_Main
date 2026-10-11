from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.buying.audit import AuditLogger
from src.buying.browser.base import BrowserContractError
from src.buying.browser.diagnostics import _mask_html, _mask_text
from src.buying.browser.playwright_client import PlaywrightIpatClient
from src.buying.config import BuyingSettings, ConfigurationError, load_settings
from src.buying.domain import (
    BetIntent,
    BetType,
    BuyingMode,
    Credentials,
    PreparedPurchase,
    PurchaseReceipt,
    PurchaseState,
    RaceInfo,
    SnapshotEvidence,
)
from src.buying.ledger import Ledger
from src.buying.risk import RiskGate
from src.buying.service import PurchaseService
from src.buying.snapshot_reader import SnapshotReader
from src.buying.strategy import SimulatorStrategy

JST = ZoneInfo("Asia/Tokyo")


def make_settings(root: Path, mode: BuyingMode = BuyingMode.PAPER) -> BuyingSettings:
    env = root / ".env"
    env.write_text(
        "\n".join(
            [
                f"BUYING_MODE={mode.value}",
                "BUYING_KILL_SWITCH=false",
                "BUYING_HEADLESS=true",
                "BUYING_ALLOWED_BET_TYPES=win",
                "BUYING_MIN_BET_YEN=100",
                "BUYING_MAX_BET_YEN=1000",
                "BUYING_MAX_RACE_YEN=2000",
                "BUYING_MAX_DAY_YEN=5000",
                "BUYING_MAX_SESSION_YEN=5000",
                "BUYING_MAX_BETS_PER_RACE=3",
                f"BUYING_SCHEDULE_DIR={root / 'races'}",
                f"BUYING_ODDS_DIR={root / 'odds'}",
                f"BUYING_DATA_DIR={root / 'buying'}",
            ]
        ),
        encoding="utf-8",
    )
    return load_settings(env, require_credentials=False)


def intent(race_id: str = "202605040711", horse: int = 3, amount: int = 100) -> BetIntent:
    return BetIntent(
        race_id=race_id,
        bet_type=BetType.WIN,
        selection=(horse,),
        amount_yen=amount,
        strategy_id="test",
        reason_code="selected",
    )


def race(race_id: str = "202605040711") -> RaceInfo:
    return RaceInfo(
        race_id=race_id,
        rt_key="202609220511",
        venue_code="05",
        race_number=11,
        post_datetime=dt.datetime(2026, 9, 22, 15, 40, tzinfo=JST),
    )


def evidence(race_id: str = "202605040711") -> SnapshotEvidence:
    return SnapshotEvidence(
        race_id=race_id,
        label="2m",
        target_datetime=dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
        acquired_at=dt.datetime(2026, 9, 22, 15, 38, 2, tzinfo=JST),
        source_datetime=dt.datetime(2026, 9, 22, 15, 37, tzinfo=JST),
        source_age_seconds=60.0,
        status="saved",
    )


class FakeBrowser:
    def __init__(self, receipt: PurchaseReceipt | None = None) -> None:
        self.receipt = receipt
        self.logged_in = False
        self.cancelled = False
        self.closed = False

    def login(self, credentials: object) -> None:
        self.logged_in = True

    def get_balance_yen(self) -> int:
        return 10_000

    def prepare_win_bets(self, race_info: RaceInfo, intents: tuple[BetIntent, ...]) -> PreparedPurchase:
        return PreparedPurchase(
            token="token",
            race_id=race_info.race_id,
            intents=intents,
            displayed_total_yen=sum(item.amount_yen for item in intents),
            displayed_summary="ok",
        )

    def cancel_prepared(self, prepared: PreparedPurchase) -> None:
        self.cancelled = True

    def submit_prepared(self, prepared: PreparedPurchase) -> PurchaseReceipt:
        if self.receipt is not None:
            return self.receipt
        return PurchaseReceipt(
            accepted=True,
            unknown=False,
            receipt_number="1234",
            accepted_at=dt.datetime.now(JST),
            amount_yen=prepared.displayed_total_yen,
            summary="accepted",
        )

    def close(self) -> None:
        self.closed = True


class ConfigTests(unittest.TestCase):
    def test_live_requires_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = root / ".env"
            env.write_text("BUYING_MODE=live\n", encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_settings(env)

    def test_amount_limits_require_100_yen_units(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            env = root / ".env"
            env.write_text(
                "BUYING_MODE=paper\nBUYING_MAX_BET_YEN=150\n",
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_settings(env, require_credentials=False)


class StrategyTests(unittest.TestCase):
    def test_simulator_rows_are_filtered_and_rounded_down(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "202605040711", "horse_number": 3, "skip_reason": "", "odds_used": 8.2},
                {"race_id": "202605040711", "horse_number": 4, "skip_reason": "ev_below_threshold"},
                {"race_id": "202605040711", "horse_number": 5, "skip_reason": ""},
            ]
        )
        result = SimulatorStrategy().from_frame(frame, target_date=dt.date(2026, 9, 22))
        self.assertEqual(len(result.intents), 2)
        self.assertEqual(result.intents[0].amount_yen, 100)
        self.assertEqual(result.intents[0].selection, (3,))
        self.assertEqual(result.intents[0].expected_odds, 8.2)


class SnapshotReaderTests(unittest.TestCase):
    def test_reads_schedule_and_saved_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            races_dir, odds_dir = root / "races", root / "odds"
            races_dir.mkdir()
            (odds_dir / "20260922").mkdir(parents=True)
            pd.DataFrame(
                [{
                    "race_id": "202605040711", "rt_key": "202609220511",
                    "venue_code": "05", "race_number": 11,
                    "post_datetime": "2026-09-22T15:40:00", "race_name": "test",
                }]
            ).to_csv(races_dir / "20260922_jra_today_schedule.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [{
                    "race_id": "202605040711", "snapshot_label": "2m",
                    "target_datetime": "2026-09-22T15:35:00",
                    "acquired_at": "2026-09-22T15:35:02",
                    "latest_happyo_datetime": "2026-09-22T15:34:00",
                    "source_age_seconds": 60, "status": "saved",
                }]
            ).to_csv(
                odds_dir / "20260922" / "20260922_scheduler_events.csv",
                index=False, encoding="utf-8-sig",
            )
            reader = SnapshotReader(races_dir, odds_dir)
            races = reader.load_races(dt.date(2026, 9, 22))
            events = reader.load_evidence(dt.date(2026, 9, 22))
            self.assertEqual(races["202605040711"].race_number, 11)
            self.assertEqual(events["202605040711"].status, "saved")


class RiskTests(unittest.TestCase):
    def test_valid_plan_passes_before_safety_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            settings = make_settings(Path(raw))
            decision = RiskGate(settings).evaluate(
                (intent(),),
                race=race(),
                evidence=evidence(),
                now=dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
                account_balance_yen=1000,
                committed_day_yen=0,
                committed_session_yen=0,
            )
            self.assertTrue(decision.accepted, decision.reasons)

    def test_plan_is_rejected_after_safety_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            settings = make_settings(Path(raw))
            decision = RiskGate(settings).evaluate(
                (intent(),),
                race=race(),
                evidence=evidence(),
                now=dt.datetime(2026, 9, 22, 15, 39, tzinfo=JST),
                account_balance_yen=1000,
                committed_day_yen=0,
                committed_session_yen=0,
            )
            self.assertIn("submit_deadline_expired", decision.reasons)


class LedgerTests(unittest.TestCase):
    def test_natural_key_prevents_duplicate_across_runs(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = Ledger(Path(raw) / "ledger.sqlite3")
            first, created1 = ledger.reserve("a:live", intent(), "run1", "hash1")
            second, created2 = ledger.reserve("a:live", intent(), "run2", "hash2")
            self.assertTrue(created1)
            self.assertFalse(created2)
            self.assertEqual(first.natural_key, second.natural_key)

    def test_invalid_state_transition_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            ledger = Ledger(Path(raw) / "ledger.sqlite3")
            entry, _ = ledger.reserve("a", intent(), "run", "hash")
            with self.assertRaises(ValueError):
                ledger.transition(entry.idempotency_key, PurchaseState.ACCEPTED)


class ServiceTests(unittest.TestCase):
    def _strategy(self) -> object:
        frame = pd.DataFrame(
            [{"race_id": "202605040711", "horse_number": 3}]
        )
        return SimulatorStrategy().from_frame(frame, target_date=dt.date(2026, 9, 22))

    def test_paper_mode_does_not_create_browser(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = make_settings(root, BuyingMode.PAPER)
            made = []
            service = PurchaseService(
                settings,
                Ledger(settings.ledger_path),
                AuditLogger(settings.log_dir),
                lambda: made.append(True),
                clock=lambda: dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
            )
            report = service.execute(
                self._strategy(),
                target_date=dt.date(2026, 9, 22),
                races={race().race_id: race()},
                evidence={race().race_id: evidence()},
            )
            self.assertEqual(report.dry_run, 1)
            self.assertEqual(made, [])

    def test_live_acceptance_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = replace(
                make_settings(root),
                mode=BuyingMode.LIVE,
                credentials=Credentials("inet", "subscriber", "pin", "pars"),
            )
            browser = FakeBrowser()
            ledger = Ledger(settings.ledger_path)
            service = PurchaseService(
                settings,
                ledger,
                AuditLogger(settings.log_dir),
                lambda: browser,
                clock=lambda: dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
            )
            report = service.execute(
                self._strategy(),
                target_date=dt.date(2026, 9, 22),
                races={race().race_id: race()},
                evidence={race().race_id: evidence()},
                allow_live=True,
            )
            self.assertEqual(report.accepted, 1)
            self.assertTrue(browser.closed)

    def test_login_failure_is_reported_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = replace(
                make_settings(root),
                mode=BuyingMode.LIVE,
                credentials=Credentials("inet", "subscriber", "pin", "pars"),
            )
            browser = FakeBrowser()

            def fail_login(_credentials: object) -> None:
                raise BrowserContractError("操作要素が見つかりません")

            browser.login = fail_login  # type: ignore[method-assign]
            service = PurchaseService(
                settings,
                Ledger(settings.ledger_path),
                AuditLogger(settings.log_dir),
                lambda: browser,
                clock=lambda: dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
            )
            report = service.execute(
                self._strategy(),
                target_date=dt.date(2026, 9, 22),
                races={race().race_id: race()},
                evidence={race().race_id: evidence()},
                allow_live=True,
            )
            self.assertEqual(report.accepted, 0)
            self.assertEqual(report.rejected, 1)
            self.assertTrue(any("login:" in item for item in report.errors))
            self.assertTrue(browser.closed)

    def test_unavailable_balance_still_allows_purchase(self) -> None:
        # 非開催日などで購入限度額が読めなくても、他のリスク審査が通れば購入する。
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = replace(
                make_settings(root),
                mode=BuyingMode.LIVE,
                credentials=Credentials("inet", "subscriber", "pin", "pars"),
            )
            browser = FakeBrowser()
            browser.get_balance_yen = lambda: None  # type: ignore[method-assign]
            service = PurchaseService(
                settings,
                Ledger(settings.ledger_path),
                AuditLogger(settings.log_dir),
                lambda: browser,
                clock=lambda: dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
            )
            report = service.execute(
                self._strategy(),
                target_date=dt.date(2026, 9, 22),
                races={race().race_id: race()},
                evidence={race().race_id: evidence()},
                allow_live=True,
            )
            self.assertEqual(report.accepted, 1)

    def test_unknown_receipt_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = replace(
                make_settings(root),
                mode=BuyingMode.LIVE,
                credentials=Credentials("inet", "subscriber", "pin", "pars"),
            )
            unknown = PurchaseReceipt(False, True, None, None, 100, "unknown")
            service = PurchaseService(
                settings,
                Ledger(settings.ledger_path),
                AuditLogger(settings.log_dir),
                lambda: FakeBrowser(unknown),
                clock=lambda: dt.datetime(2026, 9, 22, 15, 38, tzinfo=JST),
            )
            report = service.execute(
                self._strategy(),
                target_date=dt.date(2026, 9, 22),
                races={race().race_id: race()},
                evidence={race().race_id: evidence()},
                allow_live=True,
            )
            self.assertEqual(report.unknown, 1)


class BalanceParsingTests(unittest.TestCase):
    """``get_balance_yen`` の画面テキスト解釈。"""

    def _parse(self, text: str) -> int | None:
        client = PlaywrightIpatClient.__new__(PlaywrightIpatClient)
        client._body_text = lambda: text  # type: ignore[method-assign]
        return client.get_balance_yen()

    def test_table_layout_is_parsed(self) -> None:
        # 即PATのメニューはラベルと金額が別セル（間はタブか改行）。
        self.assertEqual(self._parse("購入限度額\t123,400円"), 123400)
        self.assertEqual(self._parse("購入限度額\n50,000 円"), 50000)

    def test_missing_balance_row_returns_none(self) -> None:
        # 非開催日は金額行そのものが描画されない。
        self.assertIsNone(self._parse("購入可能件数\t9,000件\n発売はございません。"))

    def test_empty_amount_cell_returns_none(self) -> None:
        # 読み込み途中は単位だけが残る。件数を金額と誤読しないこと。
        self.assertIsNone(self._parse("購入限度額\t円\n購入可能件数\t9,000件"))


class DiagnosticsMaskTests(unittest.TestCase):
    SECRETS = ("abc12345", "98765432", "1234", "5678")

    def test_credential_values_are_masked(self) -> None:
        html = (
            '<input id="inetid" name="inetid" value="abc12345">'
            '<input type="password" name="pin" value="1234">'
            '<input name="pars" value="5678" type="text">'
        )
        masked = _mask_html(html, self.SECRETS)
        for secret in self.SECRETS:
            self.assertNotIn(secret, masked)

    def test_button_labels_survive_masking(self) -> None:
        # ボタン名はセレクタ修正に必要な情報なので残す（type が value の
        # 前後どちらにあっても判定できること）。
        html = '<input type="submit" value="ログイン"><input value="送信" type="button">'
        masked = _mask_html(html, self.SECRETS)
        self.assertIn('value="ログイン"', masked)
        self.assertIn('value="送信"', masked)

    def test_member_info_and_balance_are_masked(self) -> None:
        text = "加入者番号：98765432 購入限度額: 123,400 円 残高：50,000円"
        masked = _mask_text(text, ())
        self.assertNotIn("98765432", masked)
        self.assertNotIn("123,400", masked)
        self.assertNotIn("50,000", masked)


if __name__ == "__main__":
    unittest.main()
