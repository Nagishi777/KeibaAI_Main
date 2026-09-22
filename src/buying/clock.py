"""JST時刻と購入期限の計算。"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")


def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def as_jst(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=JST)
    return value.astimezone(JST)


@dataclass(frozen=True)
class RaceClock:
    now: dt.datetime
    post_datetime: dt.datetime
    decision_lead_seconds: int
    submit_deadline_lead_seconds: int

    @property
    def decision_at(self) -> dt.datetime:
        return as_jst(self.post_datetime) - dt.timedelta(seconds=self.decision_lead_seconds)

    @property
    def submit_deadline(self) -> dt.datetime:
        return as_jst(self.post_datetime) - dt.timedelta(
            seconds=self.submit_deadline_lead_seconds
        )

    @property
    def seconds_to_deadline(self) -> float:
        return (self.submit_deadline - as_jst(self.now)).total_seconds()

    @property
    def is_expired(self) -> bool:
        return self.seconds_to_deadline < 0

