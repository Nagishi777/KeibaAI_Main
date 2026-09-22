"""購入システムのドメイン型。外部サイトや pandas に依存しない。"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class BuyingMode(str, Enum):
    PAPER = "paper"
    DRY_RUN = "dry-run"
    LIVE = "live"


class BetType(str, Enum):
    WIN = "win"
    PLACE = "place"
    BRACKET_QUINELLA = "bracket_quinella"


class PurchaseState(str, Enum):
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    SUBMITTING = "SUBMITTING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    NOT_FOUND = "NOT_FOUND"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    DRY_RUN = "DRY_RUN"


@dataclass(frozen=True)
class Credentials:
    inet_id: str
    subscriber_number: str
    pin: str
    pars_number: str


@dataclass(frozen=True)
class RaceInfo:
    race_id: str
    rt_key: str
    venue_code: str
    race_number: int
    post_datetime: dt.datetime
    race_name: str = ""


@dataclass(frozen=True)
class SnapshotEvidence:
    race_id: str
    label: str
    target_datetime: dt.datetime
    acquired_at: dt.datetime
    source_datetime: dt.datetime | None
    source_age_seconds: float | None
    status: str


@dataclass(frozen=True)
class BetIntent:
    race_id: str
    bet_type: BetType
    selection: tuple[int, ...]
    amount_yen: int
    strategy_id: str
    reason_code: str
    expected_odds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_selection(self) -> tuple[int, ...]:
        if self.bet_type in {BetType.WIN, BetType.PLACE}:
            return self.selection
        return tuple(sorted(self.selection))

    def as_public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["bet_type"] = self.bet_type.value
        value["selection"] = list(self.normalized_selection())
        return value


@dataclass(frozen=True)
class PreparedPurchase:
    token: str
    race_id: str
    intents: tuple[BetIntent, ...]
    displayed_total_yen: int
    displayed_summary: str


@dataclass(frozen=True)
class PurchaseReceipt:
    accepted: bool
    unknown: bool
    receipt_number: str | None
    accepted_at: dt.datetime | None
    amount_yen: int
    summary: str
    raw_reference: str | None = None


@dataclass(frozen=True)
class RiskDecision:
    accepted: bool
    reasons: tuple[str, ...] = ()


@dataclass
class RunReport:
    run_id: str
    target_date: str
    mode: str
    intents: int = 0
    planned_races: int = 0
    accepted: int = 0
    dry_run: int = 0
    rejected: int = 0
    unknown: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)

