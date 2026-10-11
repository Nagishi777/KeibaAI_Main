import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.simulator.features import (
    DRIFT_COL,
    LIVE_DECISION_SNAPSHOT,
    ODDS_COL,
    POOL_COL,
    build_features,
    normalize_late_columns,
)
from src.simulator.realtime_loader import available_dates, load_today_snapshots

TARGET_DATE = dt.date(2026, 9, 19)

# 特徴量の構築には 60m/30m/10m/5m と判断時点（2m）が必要
ALL_LABELS = ("60m", "30m", "10m", "5m", "2m")


def _rows(race_id: str, label: str, pool: int, odds: dict[str, float]) -> list[dict]:
    return [
        {
            "race_id": race_id,
            "umaban": umaban,
            "snapshot_label": label,
            "happyo_datetime": "2026-09-19 12:00:00",
            "hyosu_total": pool,
            "odds_win": value,
        }
        for umaban, value in odds.items()
    ]


def _write(directory: Path, race_id: str, rows: list[dict]) -> None:
    folder = directory / "20260919"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        folder / f"{race_id}_tansho_realtimeodds.csv", index=False, encoding="utf-8-sig"
    )


def _full_race(race_id: str, *, pool_5m: int, pool_2m: int,
               odds_5m: dict[str, float], odds_2m: dict[str, float]) -> list[dict]:
    """全時点が揃ったレースの行を作る（60m/30m/10m は 5m と同じ値で埋める）。"""
    rows: list[dict] = []
    for label in ("60m", "30m", "10m"):
        rows += _rows(race_id, label, pool_5m, odds_5m)
    rows += _rows(race_id, "5m", pool_5m, odds_5m)
    rows += _rows(race_id, "2m", pool_2m, odds_2m)
    return rows


class LoadTodaySnapshotsTest(unittest.TestCase):
    def test_builds_features_from_race_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            # 1番は 5m→2m でオッズが半分に下がり（資金が入った）、
            # 2番は倍に上がる（資金が逃げた）。
            _write(directory, "202606040505", _full_race(
                "202606040505",
                pool_5m=1000, pool_2m=2000,
                odds_5m={"01": 4.0, "02": 4.0},
                odds_2m={"01": 2.0, "02": 8.0},
            ))
            # 5m が無いレースは予測対象から外れる
            _write(directory, "202606040506",
                   _rows("202606040506", "2m", 1000, {"01": 2.0}))

            frame = load_today_snapshots(TARGET_DATE, directory)
            features = build_features(
                normalize_late_columns(frame, LIVE_DECISION_SNAPSHOT)
            )
            dates = available_dates(directory)

        self.assertEqual(features["race_id"].unique().tolist(), ["202606040505"])
        # 賭け判断オッズは 2m 時点
        self.assertEqual(features[ODDS_COL].tolist(), [2.0, 8.0])
        # ドリフト = log(2mオッズ / 5mオッズ)。下がった馬だけが負になる。
        self.assertAlmostEqual(features[DRIFT_COL].iloc[0], np.log(0.5))
        self.assertAlmostEqual(features[DRIFT_COL].iloc[1], np.log(2.0))
        # 総票数は百円単位なので円に直す（2000 * 100）
        self.assertEqual(features[POOL_COL].tolist(), [200000.0, 200000.0])
        # 投票額 = 0.8 * pool / odds → 5m: 200,200 / 2m: 800,200
        # 増分は 1番が 600、2番が 0 なのでシェアは 1.0 / 0.0
        self.assertEqual(features["inc_share_5_late"].tolist(), [1.0, 0.0])
        self.assertEqual(dates, [TARGET_DATE])

    def test_falls_back_to_later_snapshot_when_2m_missing(self) -> None:
        """2m が取れていないレースは 3m で代用し、使った時点を記録する。"""
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            rows = _full_race(
                "202606040505",
                pool_5m=1000, pool_2m=2000,
                odds_5m={"01": 4.0, "02": 4.0},
                odds_2m={"01": 2.0, "02": 8.0},
            )
            # 2m を落として 3m を置く
            rows = [r for r in rows if r["snapshot_label"] != "2m"]
            rows += _rows("202606040505", "3m", 2000, {"01": 2.0, "02": 8.0})
            _write(directory, "202606040505", rows)

            frame = load_today_snapshots(TARGET_DATE, directory)

        self.assertEqual(
            frame["decision_snapshot_used"].unique().tolist(), ["3m"]
        )
        self.assertEqual(frame[f"odds_{LIVE_DECISION_SNAPSHOT}_raw"].tolist(),
                         [2.0, 8.0])

    def test_missing_date_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                load_today_snapshots(TARGET_DATE, Path(raw))


if __name__ == "__main__":
    unittest.main()
