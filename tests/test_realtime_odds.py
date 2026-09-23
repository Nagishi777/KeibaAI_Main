import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.scraping.realtime_odds import (
    DEFAULT_SCHEDULE_DIR,
    SnapshotJob,
    _active_poll_keys,
    build_snapshot_jobs,
    execute_jobs,
    get_today_schedule,
    prepare_schedule,
)

RACE_KEY = "202606040505_tansho"



class TodayScheduleTest(unittest.TestCase):
    @patch("src.scraping.realtime_odds.scrape_jra_today_schedule")
    def test_always_fetches_jra_schedule_for_today(self, scrape) -> None:
        target_date = dt.date(2026, 9, 19)
        scrape.return_value = pd.DataFrame([
            {
                "post_datetime": "2026-09-19T09:45",
                "race_id": "202609040501",
                "rt_key": "202609190901",
                "venue_code": "09",
                "race_number": 1,
            }
        ])

        schedule = get_today_schedule(target_date)

        scrape.assert_called_once_with(
            target_date, output_dir=DEFAULT_SCHEDULE_DIR, headless=True
        )
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule.iloc[0]["rt_key"], "202609190901")


def _job(label: str = "1m") -> SnapshotJob:
    return SnapshotJob(
        label=label,
        target_datetime=dt.datetime(2026, 9, 19, 12, 24),
        post_datetime=dt.datetime(2026, 9, 19, 12, 25),
        race_id="202606040505",
        rt_key="202609190605",
        venue_code="06",
        race_number=5,
    )


def _fetched_tansho(happyo_datetime: str) -> dict[str, pd.DataFrame]:
    return {
        "tansho": pd.DataFrame([{
            "race_id": "202606040505",
            "rt_key": "202609190605",
            "happyo_datetime": happyo_datetime,
            "acquired_at": "2026-09-19T12:24:02",
            "umaban": "01",
            "hyosu_total": 300000,
            "odds_win": 2.5,
            "ninkijun": 1,
        }]),
        "fukusho": pd.DataFrame(),
        "wakuren": pd.DataFrame(),
    }


class ExecuteJobsTest(unittest.TestCase):
    @patch("src.scraping.realtime_odds.fetch_odds_history")
    def test_uses_current_odds_and_records_freshness(self, fetch) -> None:
        fetch.return_value = (_fetched_tansho("2026-09-19 12:23:00"), [])
        with tempfile.TemporaryDirectory() as directory:
            paths, events = execute_jobs(
                MagicMock(),
                [_job()],
                output_dir=Path(directory),
                target_date=dt.date(2026, 9, 19),
                now=dt.datetime(2026, 9, 19, 12, 24, 2),
            )

            self.assertEqual(events[0]["status"], "saved")
            self.assertEqual(events[0]["odds_dataspec"], "0B31")
            self.assertEqual(events[0]["source_age_seconds"], 60.0)
            self.assertEqual(
                paths[RACE_KEY],
                Path(directory) / "20260919" / "202606040505_tansho_realtimeodds.csv",
            )
            saved = pd.read_csv(paths[RACE_KEY])
            self.assertEqual(saved.loc[0, "odds_dataspec"], "0B31")
            self.assertTrue(
                (Path(directory) / "20260919" / "20260919_scheduler_events.csv").exists()
            )
            self.assertEqual(saved.loc[0, "source_age_seconds"], 60.0)

        self.assertEqual(fetch.call_args.kwargs["dataspec"], "0B31")

    @patch("src.scraping.realtime_odds.fetch_odds_history")
    def test_marks_old_current_odds_as_stale(self, fetch) -> None:
        fetch.return_value = (_fetched_tansho("2026-09-19 12:19:00"), [])
        with tempfile.TemporaryDirectory() as directory:
            _, events = execute_jobs(
                MagicMock(),
                [_job()],
                output_dir=Path(directory),
                target_date=dt.date(2026, 9, 19),
                now=dt.datetime(2026, 9, 19, 12, 24, 2),
            )

        self.assertEqual(events[0]["status"], "saved_stale")
        self.assertEqual(events[0]["source_age_seconds"], 300.0)

    @patch("src.scraping.realtime_odds.fetch_odds_history")
    def test_uses_prefetched_value_when_latest_record_is_after_target(self, fetch) -> None:
        histories = _fetched_tansho("2026-09-19 12:23:00")
        future = _fetched_tansho("2026-09-19 12:25:00")
        future["tansho"].loc[0, "hyosu_total"] = 999999
        fetch.return_value = (future, [])

        with tempfile.TemporaryDirectory() as directory:
            paths, events = execute_jobs(
                MagicMock(),
                [_job()],
                output_dir=Path(directory),
                target_date=dt.date(2026, 9, 19),
                now=dt.datetime(2026, 9, 19, 12, 24, 2),
                histories=histories,
            )
            saved = pd.read_csv(paths[RACE_KEY])

        self.assertEqual(events[0]["status"], "saved")
        self.assertEqual(saved.loc[0, "hyosu_total"], 300000)
        self.assertEqual(saved.loc[0, "happyo_datetime"], "2026-09-19 12:23:00")

    @patch("src.scraping.realtime_odds.fetch_odds_history")
    def test_accumulates_all_snapshots_of_a_race_in_one_file(self, fetch) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for label, minutes, happyo in (
                ("5m", 5, "2026-09-19 12:19:30"),
                ("2m", 2, "2026-09-19 12:22:30"),
                ("1m", 1, "2026-09-19 12:23:30"),
            ):
                fetch.return_value = (_fetched_tansho(happyo), [])
                job = SnapshotJob(
                    label=label,
                    target_datetime=dt.datetime(2026, 9, 19, 12, 25 - minutes),
                    post_datetime=dt.datetime(2026, 9, 19, 12, 25),
                    race_id="202606040505",
                    rt_key="202609190605",
                    venue_code="06",
                    race_number=5,
                )
                paths, _ = execute_jobs(
                    MagicMock(),
                    [job],
                    output_dir=Path(directory),
                    target_date=dt.date(2026, 9, 19),
                    now=job.target_datetime + dt.timedelta(seconds=2),
                )
            saved = pd.read_csv(paths[RACE_KEY], dtype={"umaban": str})

        self.assertEqual(saved["snapshot_label"].tolist(), ["5m", "2m", "1m"])
        self.assertEqual(saved["umaban"].tolist(), ["01", "01", "01"])

    def test_builds_jobs_from_60m_to_10s_including_2m_to_4m(self) -> None:
        schedule = prepare_schedule(
            pd.DataFrame([{
                "post_datetime": "2026-09-19T12:25",
                "race_id": "202606040505",
                "rt_key": "202609190605",
            }]),
            dt.date(2026, 9, 19),
        )

        jobs = build_snapshot_jobs(schedule)

        self.assertEqual(
            [job.label for job in jobs],
            ["60m", "30m", "10m", "5m", "4m", "3m", "2m", "1m", "10s"],
        )
        self.assertEqual(
            {job.label: job.target_datetime for job in jobs}["3m"],
            dt.datetime(2026, 9, 19, 12, 22),
        )

    def test_polling_selects_only_jobs_in_lookback_window(self) -> None:
        jobs = [
            _job("1m"),
            SnapshotJob(
                label="10s",
                target_datetime=dt.datetime(2026, 9, 19, 12, 24, 50),
                post_datetime=dt.datetime(2026, 9, 19, 12, 25),
                race_id="202606040505",
                rt_key="202609190605",
                venue_code="06",
                race_number=5,
            ),
            SnapshotJob(
                label="5m",
                target_datetime=dt.datetime(2026, 9, 19, 12, 40),
                post_datetime=dt.datetime(2026, 9, 19, 12, 45),
                race_id="202609040506",
                rt_key="202609190906",
                venue_code="09",
                race_number=6,
            ),
        ]

        keys = _active_poll_keys(
            jobs,
            0,
            dt.datetime(2026, 9, 19, 12, 23),
            lookback_seconds=120,
            collection_delay_seconds=2,
        )

        self.assertEqual(keys, ["202609190605"])


if __name__ == "__main__":
    unittest.main()
