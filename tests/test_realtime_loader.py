import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.simulator.features import FEATURE_COLS, POOL_COL, attach_features
from src.simulator.realtime_loader import available_dates, load_today_snapshots

TARGET_DATE = dt.date(2026, 9, 19)


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


class LoadTodaySnapshotsTest(unittest.TestCase):
    def test_builds_10m_5m_features_from_race_files(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            _write(directory, "202606040505", (
                _rows("202606040505", "10m", 1000, {"01": 2.0, "02": 4.0})
                + _rows("202606040505", "5m", 2000, {"01": 2.0, "02": 8.0})
                + _rows("202606040505", "1m", 9000, {"01": 1.1, "02": 90.0})
            ))
            # 5m が無いレースは予測対象から外れる
            _write(directory, "202606040506", _rows("202606040506", "10m", 1000, {"01": 2.0}))

            raw_frame = load_today_snapshots(TARGET_DATE, directory)
            features = attach_features(raw_frame)
            dates = available_dates(directory)

        self.assertEqual(FEATURE_COLS, ("odds_5m", "inc_share_10m_5m"))
        self.assertEqual(features["race_id"].unique().tolist(), ["202606040505"])
        self.assertEqual(features["odds_5m"].tolist(), [2.0, 8.0])
        # 投票額 = 0.8 * pool / odds → 10m: 400, 200 / 5m: 800, 200
        self.assertEqual(features["inc_share_10m_5m"].tolist(), [1.0, 0.0])
        self.assertEqual(features[POOL_COL].tolist(), [200000.0, 200000.0])
        self.assertEqual(dates, [TARGET_DATE])

    def test_missing_date_folder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                load_today_snapshots(TARGET_DATE, Path(raw))


if __name__ == "__main__":
    unittest.main()
