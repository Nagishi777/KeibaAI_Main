"""購入判断とは独立した、決定論的な安全審査。"""

from __future__ import annotations

import datetime as dt

from src.buying.clock import RaceClock
from src.buying.config import BuyingSettings
from src.buying.domain import BetIntent, RaceInfo, RiskDecision, SnapshotEvidence
from src.buying.snapshot_reader import DECISION_SNAPSHOT_LABEL, snapshot_offset


class RiskGate:
    def __init__(self, settings: BuyingSettings) -> None:
        self.settings = settings

    def evaluate(
        self,
        intents: tuple[BetIntent, ...],
        *,
        race: RaceInfo,
        evidence: SnapshotEvidence | None,
        now: dt.datetime,
        account_balance_yen: int | None,
        committed_day_yen: int,
        committed_session_yen: int,
    ) -> RiskDecision:
        reasons: list[str] = []
        if self.settings.kill_switch:
            reasons.append("kill_switch")
        if not intents:
            reasons.append("no_intents")
            return RiskDecision(False, tuple(reasons))
        if any(intent.race_id != race.race_id for intent in intents):
            reasons.append("race_id_mismatch")
        if len(intents) > self.settings.max_bets_per_race:
            reasons.append("max_bets_per_race")
        total = sum(intent.amount_yen for intent in intents)
        if total > self.settings.max_race_yen:
            reasons.append("max_race_yen")
        if committed_day_yen + total > self.settings.max_day_yen:
            reasons.append("max_day_yen")
        if committed_session_yen + total > self.settings.max_session_yen:
            reasons.append("max_session_yen")
        if account_balance_yen is not None and total > account_balance_yen:
            reasons.append("insufficient_balance")
        for intent in intents:
            if intent.bet_type.value not in self.settings.allowed_bet_types:
                reasons.append(f"bet_type_not_allowed:{intent.bet_type.value}")
            if intent.amount_yen % 100:
                reasons.append("amount_not_100_yen_unit")
            if not self.settings.min_bet_yen <= intent.amount_yen <= self.settings.max_bet_yen:
                reasons.append("bet_amount_out_of_range")
            if len(intent.selection) != 1 or not 1 <= intent.selection[0] <= 18:
                reasons.append("invalid_win_selection")

        race_clock = RaceClock(
            now=now,
            post_datetime=race.post_datetime,
            decision_lead_seconds=self.settings.decision_lead_seconds,
            submit_deadline_lead_seconds=self.settings.submit_deadline_lead_seconds,
        )
        if race_clock.is_expired:
            reasons.append("submit_deadline_expired")
        if evidence is None:
            reasons.append("snapshot_evidence_missing")
        else:
            if evidence.race_id != race.race_id:
                reasons.append("snapshot_race_mismatch")
            if evidence.status != "saved":
                reasons.append(f"snapshot_status:{evidence.status}")
            if evidence.label != DECISION_SNAPSHOT_LABEL:
                reasons.append("snapshot_label_mismatch")
            expected_target = race.post_datetime - snapshot_offset(DECISION_SNAPSHOT_LABEL)
            target_skew = abs((evidence.target_datetime - expected_target).total_seconds())
            if target_skew > self.settings.max_clock_skew_seconds:
                reasons.append("snapshot_target_time_mismatch")
            if evidence.acquired_at > now + dt.timedelta(
                seconds=self.settings.max_clock_skew_seconds
            ):
                reasons.append("snapshot_acquired_in_future")
            if evidence.source_datetime is None:
                reasons.append("source_datetime_missing")
            elif evidence.source_datetime > evidence.target_datetime:
                reasons.append("source_after_target")
            if (
                evidence.source_age_seconds is None
                or evidence.source_age_seconds < 0
                or evidence.source_age_seconds > self.settings.max_odds_age_seconds
            ):
                reasons.append("odds_stale")
        return RiskDecision(not reasons, tuple(dict.fromkeys(reasons)))
