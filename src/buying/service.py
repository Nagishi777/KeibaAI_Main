"""判断、リスク審査、画面操作、台帳を結ぶ購入ユースケース。"""

from __future__ import annotations

import datetime as dt
import uuid
from collections import defaultdict
from typing import Callable

from src.buying.audit import AuditLogger
from src.buying.browser.base import IpatClient
from src.buying.clock import now_jst
from src.buying.config import BuyingSettings
from src.buying.domain import BuyingMode, PurchaseState, RaceInfo, RunReport, SnapshotEvidence
from src.buying.ledger import Ledger, LedgerEntry
from src.buying.risk import RiskGate
from src.buying.strategy import StrategyResult


class LiveModeNotAuthorized(PermissionError):
    pass


class PurchaseService:
    def __init__(
        self,
        settings: BuyingSettings,
        ledger: Ledger,
        audit: AuditLogger,
        browser_factory: Callable[[], IpatClient],
        *,
        clock: Callable[[], dt.datetime] = now_jst,
    ) -> None:
        self.settings = settings
        self.ledger = ledger
        self.audit = audit
        self.browser_factory = browser_factory
        self.clock = clock
        self.risk = RiskGate(settings)

    def execute(
        self,
        strategy: StrategyResult,
        *,
        target_date: dt.date,
        races: dict[str, RaceInfo],
        evidence: dict[str, SnapshotEvidence],
        allow_live: bool = False,
    ) -> RunReport:
        if self.settings.mode == BuyingMode.LIVE and not allow_live:
            raise LiveModeNotAuthorized(
                "live実行には BUYING_MODE=live に加えて --live が必要です"
            )
        report = RunReport(
            run_id=uuid.uuid4().hex,
            target_date=target_date.isoformat(),
            mode=self.settings.mode.value,
            intents=len(strategy.intents),
        )
        grouped: dict[str, list] = defaultdict(list)
        for intent in strategy.intents:
            grouped[intent.race_id].append(intent)
        report.planned_races = len(grouped)
        ledger_alias = f"{self.settings.account_alias}:{self.settings.mode.value}"
        pending: dict[str, list] = {}
        for race_id, intents in grouped.items():
            existing = [
                self.ledger.find_natural(ledger_alias, item, strategy.decision_run_id)
                for item in intents
            ]
            if any(item is not None for item in existing):
                report.skipped_existing += len(intents)
            else:
                pending[race_id] = intents
        grouped = pending
        self.audit.write(
            "run_started",
            run_id=report.run_id,
            decision_run_id=strategy.decision_run_id,
            input_hash=strategy.input_hash,
            mode=self.settings.mode.value,
            intents=len(strategy.intents),
        )

        browser: IpatClient | None = None
        balance: int | None = None
        session_committed = 0
        try:
            if self.settings.mode != BuyingMode.PAPER and not self.settings.kill_switch and grouped:
                if self.settings.credentials is None:
                    raise ValueError("ブラウザ実行に必要な認証情報がありません")
                browser = self.browser_factory()
                browser.login(self.settings.credentials)
                balance = browser.get_balance_yen()
                self.audit.write("browser_logged_in", run_id=report.run_id, balance_yen=balance)

            for race_id, raw_intents in sorted(grouped.items()):
                intents = tuple(raw_intents)
                race = races.get(race_id)
                if race is None:
                    report.rejected += len(intents)
                    self.audit.write("race_rejected", race_id=race_id, reasons=["race_not_found"])
                    continue
                committed_day = self.ledger.committed_amount(target_date)
                decision = self.risk.evaluate(
                    intents,
                    race=race,
                    evidence=evidence.get(race_id),
                    now=self.clock(),
                    account_balance_yen=balance,
                    committed_day_yen=committed_day,
                    committed_session_yen=session_committed,
                )
                if not decision.accepted:
                    report.rejected += len(intents)
                    self.audit.write(
                        "race_rejected", race_id=race_id, reasons=list(decision.reasons)
                    )
                    continue
                entries, all_new = self._reserve_all(intents, strategy, target_date)
                if not all_new:
                    report.skipped_existing += len(intents)
                    self.audit.write("race_skipped_existing", race_id=race_id)
                    continue
                for entry in entries:
                    self.ledger.transition(entry.idempotency_key, PurchaseState.VALIDATED)

                if self.settings.mode == BuyingMode.PAPER:
                    for entry in entries:
                        self.ledger.transition(entry.idempotency_key, PurchaseState.DRY_RUN)
                    report.dry_run += len(intents)
                    self.audit.write("paper_validated", race_id=race_id, total_yen=sum(i.amount_yen for i in intents))
                    continue

                assert browser is not None
                try:
                    prepared = browser.prepare_win_bets(race, intents)
                    expected_total = sum(intent.amount_yen for intent in intents)
                    if prepared.displayed_total_yen != expected_total:
                        raise ValueError(
                            f"確認画面総額不一致: expected={expected_total}, "
                            f"actual={prepared.displayed_total_yen}"
                        )
                    if self.settings.mode == BuyingMode.DRY_RUN:
                        browser.cancel_prepared(prepared)
                        for entry in entries:
                            self.ledger.transition(entry.idempotency_key, PurchaseState.DRY_RUN)
                        report.dry_run += len(intents)
                        self.audit.write("dry_run_completed", race_id=race_id, total_yen=expected_total)
                        continue

                    for entry in entries:
                        self.ledger.transition(entry.idempotency_key, PurchaseState.SUBMITTING)
                    receipt = browser.submit_prepared(prepared)
                    if receipt.accepted and receipt.amount_yen == expected_total:
                        for entry in entries:
                            self.ledger.transition(
                                entry.idempotency_key,
                                PurchaseState.ACCEPTED,
                                receipt_number=receipt.receipt_number,
                                receipt_summary=receipt.summary,
                            )
                        report.accepted += len(intents)
                        session_committed += expected_total
                        if balance is not None:
                            balance -= expected_total
                        self.audit.write(
                            "purchase_accepted",
                            race_id=race_id,
                            total_yen=expected_total,
                            receipt_number=receipt.receipt_number,
                        )
                    else:
                        self._mark_unknown(entries, receipt.summary)
                        report.unknown += len(intents)
                        self.audit.write("purchase_unknown", race_id=race_id, detail=receipt.summary)
                        if self.settings.mode == BuyingMode.LIVE:
                            break
                except Exception as exc:
                    current = [self._state(entry.idempotency_key) for entry in entries]
                    if PurchaseState.SUBMITTING.value in current:
                        self._mark_unknown(entries, str(exc))
                        report.unknown += len(intents)
                    else:
                        for entry in entries:
                            if self._state(entry.idempotency_key) == PurchaseState.VALIDATED.value:
                                self.ledger.transition(
                                    entry.idempotency_key,
                                    PurchaseState.REJECTED,
                                    error=str(exc),
                                )
                        report.rejected += len(intents)
                    report.errors.append(f"{race_id}: {exc}")
                    self.audit.write("race_error", race_id=race_id, error=str(exc))
                    if self.settings.mode == BuyingMode.LIVE:
                        break
        finally:
            if browser is not None:
                browser.close()
            self.audit.write("run_finished", **report.__dict__)
        return report

    def _reserve_all(
        self, intents: tuple, strategy: StrategyResult, target_date: dt.date
    ) -> tuple[list[LedgerEntry], bool]:
        entries: list[LedgerEntry] = []
        all_new = True
        for intent in intents:
            entry, created = self.ledger.reserve(
                f"{self.settings.account_alias}:{self.settings.mode.value}",
                intent,
                strategy.decision_run_id,
                strategy.input_hash,
                target_date,
            )
            entries.append(entry)
            all_new = all_new and created
        if not all_new:
            for entry, intent in zip(entries, intents):
                if entry.state == PurchaseState.PLANNED.value:
                    self.ledger.transition(
                        entry.idempotency_key,
                        PurchaseState.REJECTED,
                        error="same race contains an existing purchase",
                    )
        return entries, all_new

    def _mark_unknown(self, entries: list[LedgerEntry], detail: str) -> None:
        for entry in entries:
            state = self._state(entry.idempotency_key)
            if state == PurchaseState.SUBMITTING.value:
                self.ledger.transition(
                    entry.idempotency_key, PurchaseState.UNKNOWN, error=detail
                )

    def _state(self, idempotency_key: str) -> str:
        # 公開APIを小さく保つため、冪等な同状態遷移で現在値を取得する代わりに
        # 台帳のunknown一覧と直接照合せず、読み取り専用メソッドを利用する。
        return self.ledger.get(idempotency_key).state
