import datetime as dt
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.buying.config import ConfigurationError
from src.pipeline.daily import (
    DailyPipeline,
    PipelineOptions,
    RaceDecisionWorker,
    decision_race_ids,
    sleep_until,
)
from src.scraping.realtime_odds import prepare_schedule, run_scheduler
from src.simulator.predict_today import save_outputs


def _event(race_id: str, label: str, status: str = "saved") -> dict:
    return {"race_id": race_id, "snapshot_label": label, "status": status}


class DecisionRaceIdsTest(unittest.TestCase):
    def test_only_saved_5m_events_trigger_decisions(self) -> None:
        events = [
            _event("R1", "10m"),
            _event("R2", "5m"),
            _event("R3", "5m", "saved_stale"),
            _event("R4", "5m", "no_odds_before_target"),
            _event("R2", "5m"),
        ]

        self.assertEqual(decision_race_ids(events), ["R2", "R3"])


class RaceDecisionWorkerTest(unittest.TestCase):
    def test_batches_waiting_races_and_ignores_duplicates(self) -> None:
        release = threading.Event()
        calls: list[list[str]] = []

        def decide(race_ids: list[str]) -> dict:
            calls.append(list(race_ids))
            release.wait(5)
            return {"race_ids": race_ids}

        worker = RaceDecisionWorker(decide)
        worker.start()
        worker.submit(["R1"])
        # R1 の処理中に届いたレースは次の1回にまとめて処理される
        while not calls:
            threading.Event().wait(0.01)
        worker.submit(["R2", "R3", "R1"])
        worker.submit(["R3"])
        release.set()
        worker.stop()

        self.assertEqual(calls, [["R1"], ["R2", "R3"]])

    def test_records_errors_and_keeps_running(self) -> None:
        def decide(race_ids: list[str]) -> dict:
            if race_ids == ["BAD"]:
                raise ValueError("5m が無い")
            return {"race_ids": race_ids}

        recorded: list[dict] = []
        worker = RaceDecisionWorker(decide, on_result=recorded.append)
        worker.start()
        worker.submit(["BAD"])
        worker.stop()
        worker2 = RaceDecisionWorker(decide, on_result=recorded.append)
        worker2.start()
        worker2.submit(["OK"])
        worker2.stop()

        self.assertIn("5m が無い", recorded[0]["error"])
        self.assertEqual(recorded[1], {"race_ids": ["OK"]})


class SleepUntilTest(unittest.TestCase):
    def test_waits_in_steps_until_target(self) -> None:
        clock = [dt.datetime(2026, 9, 19, 16, 0)]
        slept: list[float] = []

        def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            clock[0] += dt.timedelta(seconds=seconds)

        sleep_until(
            dt.datetime(2026, 9, 19, 16, 2, 30),
            now=lambda: clock[0], sleep=fake_sleep,
        )

        self.assertEqual(slept, [60.0, 60.0, 30.0])


class RunSchedulerCallbackTest(unittest.TestCase):
    @patch("src.scraping.realtime_odds.create_jvlink")
    @patch("src.scraping.realtime_odds.fetch_odds_history")
    @patch("src.scraping.realtime_odds._now_jst_naive")
    def test_notifies_events_and_reports_completion(self, now, fetch, _create) -> None:
        now.return_value = dt.datetime(2026, 9, 19, 13, 0)
        fetch.return_value = ({"tansho": pd.DataFrame(), "fukusho": pd.DataFrame(),
                               "wakuren": pd.DataFrame()}, [])
        schedule = prepare_schedule(
            pd.DataFrame([{
                "post_datetime": "2026-09-19T12:25",
                "race_id": "202606040505",
                "rt_key": "202609190605",
            }]),
            dt.date(2026, 9, 19),
        )
        received: list[list[dict]] = []

        with tempfile.TemporaryDirectory() as directory:
            completed = run_scheduler(
                schedule, dt.date(2026, 9, 19), output_dir=Path(directory),
                include_past=True, on_jobs_executed=received.append,
            )

        self.assertTrue(completed)
        self.assertEqual(len(received), 1)
        self.assertEqual(
            sorted(event["snapshot_label"] for event in received[0]),
            sorted(["60m", "30m", "10m", "5m", "4m", "3m", "2m", "1m", "10s"]),
        )


class SaveOutputsMergeTest(unittest.TestCase):
    @staticmethod
    def _work(race_id: str, bet: bool) -> pd.DataFrame:
        return pd.DataFrame([{
            "race_id": race_id, "horse_number": 1, "pred_proba": 0.2,
            "odds_5m": 8.0, "ev": 1.6, "bet_amount": 300.0 if bet else 0.0,
            "pool_5m": 1e7, "inc_share_10m_5m": 0.1,
            "skip_reason": "" if bet else "ev_below_threshold",
        }])

    def test_per_race_predictions_are_appended_and_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            day = dt.date(2026, 9, 19)
            save_outputs(self._work("R1", True), day, out, merge_existing=True)
            save_outputs(self._work("R2", True), day, out, merge_existing=True)
            # R1 を再予測して推奨が無くなったら、R1 の古い推奨行は消える
            save_outputs(self._work("R1", False), day, out, merge_existing=True)

            all_rows = pd.read_csv(out / "20260919_all.csv", dtype={"race_id": str})
            bets = pd.read_csv(out / "20260919_bets.csv", dtype={"race_id": str})

        self.assertEqual(all_rows["race_id"].tolist(), ["R1", "R2"])
        self.assertEqual(all_rows["skip_reason"].fillna("").tolist(), ["ev_below_threshold", ""])
        self.assertEqual(bets["race_id"].tolist(), ["R2"])


class BuyingModeGateTest(unittest.TestCase):
    def test_live_mode_requires_live_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / ".env"
            env.write_text("\n".join([
                "BUYING_MODE=live",
                "JRA_INET_ID=a", "JRA_KANYUSHA_NO=b", "JRA_PIN=c", "JRA_PARS_NO=d",
            ]), encoding="utf-8")
            pipeline = DailyPipeline(
                PipelineOptions(env_path=env), target_date=dt.date(2026, 9, 19),
                output_root=Path(directory) / "output",
            )

            with self.assertRaises(ConfigurationError):
                pipeline._buying_settings()

            pipeline.options.live = True
            self.assertEqual(pipeline._buying_settings().mode.value, "live")


if __name__ == "__main__":
    unittest.main()
