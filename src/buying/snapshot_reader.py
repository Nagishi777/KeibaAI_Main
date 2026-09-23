"""既存スクレイパの開催表・取得イベントを購入用ドメインへ変換する。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from src.buying.clock import JST
from src.buying.domain import RaceInfo, SnapshotEvidence
from src.simulator.features import DECISION_SNAPSHOT

# 購入判断に使うスナップショット。シミュレータの判断時点と必ず一致させる。
DECISION_SNAPSHOT_LABEL = DECISION_SNAPSHOT


def snapshot_offset(label: str) -> dt.timedelta:
    """``5m`` / ``10s`` 形式の時点ラベルを発走時刻からの差に変換する。"""
    units = {"m": "minutes", "s": "seconds"}
    unit = units.get(label[-1:])
    if unit is None or not label[:-1].isdigit():
        raise ValueError(f"時点ラベルの形式が不正です: {label!r}")
    return dt.timedelta(**{unit: int(label[:-1])})


class SnapshotDataError(ValueError):
    pass


class SnapshotReader:
    def __init__(self, schedule_dir: Path, odds_dir: Path) -> None:
        self.schedule_dir = Path(schedule_dir)
        self.odds_dir = Path(odds_dir)

    def load_races(self, target_date: dt.date) -> dict[str, RaceInfo]:
        path = self.schedule_dir / f"{target_date:%Y%m%d}_jra_today_schedule.csv"
        if not path.exists():
            raise FileNotFoundError(f"開催表が見つかりません: {path}")
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype={"race_id": str, "rt_key": str, "venue_code": str},
        )
        required = {"race_id", "rt_key", "venue_code", "race_number", "post_datetime"}
        missing = required - set(frame.columns)
        if missing:
            raise SnapshotDataError(f"開催表に必要な列がありません: {sorted(missing)}")
        posts = pd.to_datetime(frame["post_datetime"], errors="coerce")
        if posts.isna().any():
            raise SnapshotDataError("開催表の post_datetime に不正値があります")
        races: dict[str, RaceInfo] = {}
        for row, post in zip(frame.to_dict("records"), posts):
            race_id = str(row["race_id"]).zfill(12)
            if race_id in races:
                raise SnapshotDataError(f"開催表の race_id が重複しています: {race_id}")
            post_dt = post.to_pydatetime()
            if post_dt.tzinfo is None:
                post_dt = post_dt.replace(tzinfo=JST)
            else:
                post_dt = post_dt.astimezone(JST)
            races[race_id] = RaceInfo(
                race_id=race_id,
                rt_key=str(row["rt_key"]).zfill(12),
                venue_code=str(row["venue_code"]).zfill(2),
                race_number=int(row["race_number"]),
                post_datetime=post_dt,
                race_name=str(row.get("race_name", "")),
            )
        return races

    def load_evidence(
        self, target_date: dt.date, *, label: str = DECISION_SNAPSHOT_LABEL
    ) -> dict[str, SnapshotEvidence]:
        stamp = f"{target_date:%Y%m%d}"
        candidates = [
            self.odds_dir / stamp / f"{stamp}_scheduler_events.csv",
            self.odds_dir / f"{stamp}_scheduler_events.csv",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(f"取得イベントが見つかりません: {candidates}")
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
        required = {"race_id", "snapshot_label", "target_datetime", "acquired_at", "status"}
        missing = required - set(frame.columns)
        if missing:
            raise SnapshotDataError(f"取得イベントに必要な列がありません: {sorted(missing)}")
        frame = frame[frame["snapshot_label"].astype(str) == label].copy()
        evidence: dict[str, SnapshotEvidence] = {}
        for row in frame.to_dict("records"):
            race_id = str(row["race_id"]).zfill(12)
            target = self._parse_datetime(row["target_datetime"], "target_datetime")
            acquired = self._parse_datetime(row["acquired_at"], "acquired_at")
            source = self._optional_datetime(row.get("latest_happyo_datetime"))
            age = pd.to_numeric(row.get("source_age_seconds"), errors="coerce")
            item = SnapshotEvidence(
                race_id=race_id,
                label=label,
                target_datetime=target,
                acquired_at=acquired,
                source_datetime=source,
                source_age_seconds=None if pd.isna(age) else float(age),
                status=str(row["status"]),
            )
            old = evidence.get(race_id)
            if old is None or item.acquired_at > old.acquired_at:
                evidence[race_id] = item
        return evidence

    @staticmethod
    def _parse_datetime(value: object, column: str) -> dt.datetime:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise SnapshotDataError(f"{column} に不正値があります: {value!r}")
        result = parsed.to_pydatetime()
        return result.replace(tzinfo=JST) if result.tzinfo is None else result.astimezone(JST)

    @classmethod
    def _optional_datetime(cls, value: object) -> dt.datetime | None:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return cls._parse_datetime(value, "latest_happyo_datetime")

