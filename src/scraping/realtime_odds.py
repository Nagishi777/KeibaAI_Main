"""発走時刻に合わせて JV-Link の当日オッズをレース別 CSV に保存する。

各レースについて 60分・30分・10分・5分・4分・3分・2分・1分・10秒前のジョブを
生成する。速報オッズを基準時刻の少し前から継続取得し、
``happyo_datetime <= 基準時刻`` の値だけを採用するため、取得時点のキャッシュと
未来のオッズの混入を抑える。

出力（1レース・1券種につき1ファイル。全時点を ``snapshot_label`` 列で持つ）::

    data/processed/realtime_odds/YYYYMMDD/{race_id}_{tansho|fukusho|wakuren}_realtimeodds.csv
    data/processed/realtime_odds/YYYYMMDD/YYYYMMDD_scheduler_events.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import pandas as pd

from zoneinfo import ZoneInfo

from src.scraping._jvlink_o1 import (
    BET_TYPES,
    DATASPEC_O1_CURRENT,
    DATASPEC_O1_HISTORY,
    KEY_COLUMN,
    JVLink,
    create_jvlink,
    fetch_odds_history,
)
from src.scraping._jra_today_schedule import (
    DEFAULT_OUTPUT_DIR as DEFAULT_SCHEDULE_DIR,
    scrape_jra_today_schedule,
)

logger = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")

DEFAULT_OUTPUT_DIR = Path("data/processed/realtime_odds")
DEFAULT_POLL_INTERVAL_SECONDS = 30.0
DEFAULT_POLL_LOOKBACK_SECONDS = 120.0
# 当回レースの公式な最長更新間隔（120秒）に余裕を持たせる。
NEAR_RACE_FRESHNESS_LIMIT_SECONDS = 180.0
# 当回以外は更新に最大710秒程度かかるため、早い時点は許容幅を広げる。
EARLY_FRESHNESS_LIMIT_SECONDS = 900.0
DEFAULT_SNAPSHOT_SPECS = (
    ("60m", dt.timedelta(minutes=60)),
    ("30m", dt.timedelta(minutes=30)),
    ("10m", dt.timedelta(minutes=10)),
    ("5m", dt.timedelta(minutes=5)),
    ("4m", dt.timedelta(minutes=4)),
    ("3m", dt.timedelta(minutes=3)),
    ("2m", dt.timedelta(minutes=2)),
    ("1m", dt.timedelta(minutes=1)),
    ("10s", dt.timedelta(seconds=10)),
)
# 同一時刻に期限となったジョブは発走に近い時点から処理する。
SNAPSHOT_PRIORITY = {
    label: priority
    for priority, (label, _) in enumerate(
        sorted(DEFAULT_SNAPSHOT_SPECS, key=lambda spec: spec[1])
    )
}
REQUIRED_SCHEDULE_COLUMNS = {"post_datetime", "race_id", "rt_key"}
REALTIME_ODDS_SUFFIX = "realtimeodds"
# 同一プロセス内で CSV を読む処理（src.pipeline の予測スレッド）と書込みを排他する。
CSV_LOCK = threading.RLock()
# ジョブ群の実行後に取得イベント（execute_jobs の events）を受け取るコールバック。
JobsExecutedCallback = Callable[[list[dict[str, object]]], None]


@dataclass(frozen=True)
class SnapshotJob:
    """1レース・1基準時刻分の取得ジョブ。時刻は日本時間の naive datetime。"""

    label: str
    target_datetime: dt.datetime
    post_datetime: dt.datetime
    race_id: str
    rt_key: str
    venue_code: str
    race_number: int

    @property
    def sort_key(self) -> tuple[dt.datetime, int, str]:
        return (self.target_datetime, SNAPSHOT_PRIORITY[self.label], self.rt_key)


def _now_jst_naive() -> dt.datetime:
    return dt.datetime.now(JST).replace(tzinfo=None)


def prepare_schedule(frame: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
    """取得した当日出馬表を検証し、スケジューラ用に整形する。"""
    missing = REQUIRED_SCHEDULE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"取得した出馬表に必要な列がありません: {sorted(missing)}")

    frame = frame.copy()
    parsed = pd.to_datetime(frame["post_datetime"], errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    if parsed.isna().any():
        raise ValueError(
            "取得した出馬表の post_datetime が不正です: "
            f"{frame.loc[parsed.isna(), 'post_datetime'].head(5).tolist()}"
        )
    frame["post_datetime"] = parsed
    frame["race_id"] = frame["race_id"].astype(str).str.zfill(12)
    frame["rt_key"] = frame["rt_key"].astype(str).str.zfill(12)
    if "venue_code" not in frame.columns:
        frame["venue_code"] = frame["rt_key"].str[8:10]
    frame["venue_code"] = frame["venue_code"].astype(str).str.zfill(2)
    if "race_number" not in frame.columns:
        frame["race_number"] = frame["rt_key"].str[10:12]
    frame["race_number"] = pd.to_numeric(frame["race_number"], errors="raise").astype(int)

    selected = frame[frame["post_datetime"].dt.date == target_date].copy()
    if selected.empty:
        raise ValueError(f"取得した出馬表に当日のレースがありません: {target_date}")
    if selected["rt_key"].duplicated().any():
        raise ValueError("取得した出馬表の rt_key が重複しています")
    return selected.sort_values(["post_datetime", "rt_key"], kind="stable").reset_index(drop=True)


def get_today_schedule(
    target_date: dt.date,
    *,
    headless: bool = True,
    schedule_dir: Path = DEFAULT_SCHEDULE_DIR,
) -> pd.DataFrame:
    """JRA出馬表を毎回取得し、当日の発走時刻一覧を返す。

    取得結果は ``schedule_dir/YYYYMMDD_jra_today_schedule.csv`` にも保存される。
    """
    logger.info("当日の発走時刻をJRA出馬表から取得します: %s", target_date)
    frame = scrape_jra_today_schedule(
        target_date, output_dir=Path(schedule_dir), headless=headless
    )
    if frame.empty:
        raise RuntimeError("当日の発走時刻を取得できませんでした")
    return prepare_schedule(frame, target_date)


def build_snapshot_jobs(
    schedule: pd.DataFrame,
    specs: Iterable[tuple[str, dt.timedelta]] = DEFAULT_SNAPSHOT_SPECS,
) -> list[SnapshotJob]:
    """発走時刻一覧から時点別の取得ジョブを生成する。"""
    jobs: list[SnapshotJob] = []
    for row in schedule.itertuples(index=False):
        post_datetime = row.post_datetime.to_pydatetime()
        for label, offset in specs:
            jobs.append(SnapshotJob(
                label=label, target_datetime=post_datetime - offset, post_datetime=post_datetime,
                race_id=str(row.race_id), rt_key=str(row.rt_key),
                venue_code=str(row.venue_code), race_number=int(row.race_number),
            ))
    return sorted(jobs, key=lambda job: job.sort_key)


def _select_snapshot(
    history: pd.DataFrame,
    job: SnapshotJob,
    bet_type: str,
    *,
    dataspec: str,
) -> pd.DataFrame:
    """基準時刻以前で最も新しい O1 発表値をレース内の組番ごとに選ぶ。"""
    if history.empty:
        return history.copy()
    key_column = KEY_COLUMN[bet_type]
    frame = history[history["rt_key"].astype(str) == job.rt_key].copy()
    if frame.empty:
        return frame
    frame["happyo_datetime"] = pd.to_datetime(frame["happyo_datetime"], errors="coerce")
    frame = frame[
        frame["happyo_datetime"].notna()
        & (frame["happyo_datetime"] <= pd.Timestamp(job.target_datetime))
    ]
    if frame.empty:
        return frame
    frame["_record_order"] = range(len(frame))
    frame = frame.sort_values(
        ["race_id", key_column, "happyo_datetime", "_record_order"], kind="stable"
    ).drop_duplicates(["race_id", key_column], keep="last")
    frame = frame.drop(columns="_record_order")
    frame["race_id"] = job.race_id
    frame["venue_code"] = job.venue_code
    frame["race_number"] = job.race_number
    frame["post_datetime"] = job.post_datetime.isoformat(timespec="seconds")
    frame["snapshot_label"] = job.label
    frame["target_datetime"] = job.target_datetime.isoformat(timespec="seconds")
    frame["source_age_seconds"] = (
        pd.Timestamp(job.target_datetime) - frame["happyo_datetime"]
    ).dt.total_seconds()
    frame["odds_dataspec"] = dataspec
    return frame.reset_index(drop=True)


def _empty_histories() -> dict[str, pd.DataFrame]:
    return {bet_type: pd.DataFrame() for bet_type in BET_TYPES}


def _append_histories(
    histories: dict[str, pd.DataFrame],
    fetched: dict[str, pd.DataFrame],
) -> None:
    """複数回取得した速報値を取得順を保ってメモリ上に蓄積する。"""
    for bet_type in BET_TYPES:
        frame = fetched.get(bet_type, pd.DataFrame())
        if frame.empty:
            continue
        existing = histories.get(bet_type, pd.DataFrame())
        histories[bet_type] = pd.concat(
            [existing, frame], ignore_index=True, sort=False
        )


def _fetch_into_histories(
    jv: JVLink,
    rt_keys: Iterable[str],
    histories: dict[str, pd.DataFrame],
    *,
    acquired_at: dt.datetime,
    dataspec: str,
) -> list[str]:
    keys = sorted(set(rt_keys))
    if not keys:
        return []
    fetched, failed_keys = fetch_odds_history(
        jv, keys, acquired_at=acquired_at, dataspec=dataspec
    )
    _append_histories(histories, fetched)
    return failed_keys


def _freshness_limit_seconds(job: SnapshotJob) -> float:
    if job.label in {"60m", "30m"}:
        return EARLY_FRESHNESS_LIMIT_SECONDS
    return NEAR_RACE_FRESHNESS_LIMIT_SECONDS


def _active_poll_keys(
    jobs: list[SnapshotJob],
    start_index: int,
    now: dt.datetime,
    *,
    lookback_seconds: float,
    collection_delay_seconds: float,
) -> list[str]:
    """現在が先行取得区間に入っている未実行レースのキーを返す。"""
    lookback = dt.timedelta(seconds=lookback_seconds)
    delay = dt.timedelta(seconds=collection_delay_seconds)
    keys: set[str] = set()
    for job in jobs[start_index:]:
        window_start = job.target_datetime - lookback
        if window_start > now:
            break
        if now <= job.target_datetime + delay:
            keys.add(job.rt_key)
    return sorted(keys)


def _upsert_csv(
    path: Path,
    frame: pd.DataFrame,
    dedupe_columns: list[str],
    *,
    sort_columns: Optional[list[str]] = None,
) -> Path:
    """生成済みCSVの同一レコードを更新して保存する。"""
    with CSV_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_csv(
                path,
                dtype={"race_id": str, "rt_key": str, "umaban": str, "kumiban": str},
                encoding="utf-8-sig",
            )
            frame = pd.concat([existing, frame], ignore_index=True, sort=False)
        keys = [column for column in dedupe_columns if column in frame.columns]
        if keys:
            frame = frame.drop_duplicates(keys, keep="last")
        order = [column for column in (sort_columns or []) if column in frame.columns]
        if order:
            frame = frame.sort_values(order, kind="stable")
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def date_dir(output_dir: Path, target_date: dt.date) -> Path:
    """対象日の出力フォルダ ``output_dir/YYYYMMDD`` を返す。"""
    return Path(output_dir) / f"{target_date:%Y%m%d}"


def race_odds_path(
    output_dir: Path, target_date: dt.date, race_id: str, bet_type: str
) -> Path:
    """1レース・1券種分のオッズCSVのパスを返す。"""
    return date_dir(output_dir, target_date) / (
        f"{race_id}_{bet_type}_{REALTIME_ODDS_SUFFIX}.csv"
    )


def save_offset_snapshots(
    snapshots: dict[str, list[pd.DataFrame]], output_dir: Path, target_date: dt.date
) -> dict[str, Path]:
    """レース・券種ごとのCSVへ、全時点のスナップショットを重複なく追記する。

    戻り値のキーは ``{race_id}_{bet_type}``。
    """
    paths: dict[str, Path] = {}
    for bet_type, frames in snapshots.items():
        nonempty = [frame for frame in frames if not frame.empty]
        if not nonempty:
            continue
        key_column = KEY_COLUMN[bet_type]
        combined = pd.concat(nonempty, ignore_index=True, sort=False)
        for race_id, group in combined.groupby("race_id", sort=False):
            path = race_odds_path(output_dir, target_date, str(race_id), bet_type)
            _upsert_csv(
                path, group,
                ["race_id", key_column, "snapshot_label", "target_datetime"],
                sort_columns=["target_datetime", key_column],
            )
            paths[f"{race_id}_{bet_type}"] = path
            logger.info(
                "時点別オッズ保存: %s (%s, %d 件)",
                path, ",".join(group["snapshot_label"].unique()), len(group),
            )
    return paths


def _save_events(events: list[dict[str, object]], output_dir: Path, target_date: dt.date) -> Path:
    return _upsert_csv(
        date_dir(output_dir, target_date) / f"{target_date:%Y%m%d}_scheduler_events.csv",
        pd.DataFrame(events), ["rt_key", "snapshot_label", "target_datetime"],
    )


def execute_jobs(
    jv: JVLink,
    jobs: list[SnapshotJob],
    *,
    output_dir: Path,
    target_date: dt.date,
    now: Optional[dt.datetime] = None,
    dataspec: str = DATASPEC_O1_CURRENT,
    histories: Optional[dict[str, pd.DataFrame]] = None,
    check_freshness: bool = True,
) -> tuple[dict[str, Path], list[dict[str, object]]]:
    """同一タイミングで期限となったジョブ群を取得・選別・保存する。"""
    if not jobs:
        return {}, []
    acquired_at = now or _now_jst_naive()
    if histories is None:
        histories = _empty_histories()
    failed_keys = _fetch_into_histories(
        jv,
        (job.rt_key for job in jobs),
        histories,
        acquired_at=acquired_at,
        dataspec=dataspec,
    )
    snapshots: dict[str, list[pd.DataFrame]] = {bet_type: [] for bet_type in BET_TYPES}
    events: list[dict[str, object]] = []
    for job in jobs:
        rows = 0
        source_ages: list[float] = []
        happyo_datetimes: list[pd.Timestamp] = []
        for bet_type in BET_TYPES:
            selected = _select_snapshot(
                histories[bet_type], job, bet_type, dataspec=dataspec
            )
            rows += len(selected)
            if not selected.empty:
                source_ages.extend(selected["source_age_seconds"].dropna().tolist())
                happyo_datetimes.extend(
                    pd.to_datetime(selected["happyo_datetime"], errors="coerce")
                    .dropna()
                    .tolist()
                )
                snapshots[bet_type].append(selected)
        source_age_seconds = max(source_ages) if source_ages else None
        freshness_limit = _freshness_limit_seconds(job) if check_freshness else None
        if rows:
            status = (
                "saved_stale"
                if freshness_limit is not None
                and source_age_seconds is not None
                and source_age_seconds > freshness_limit
                else "saved"
            )
            if status == "saved_stale":
                logger.warning(
                    "古い速報オッズを保存: rt_key=%s label=%s age=%.0f秒 limit=%.0f秒",
                    job.rt_key,
                    job.label,
                    source_age_seconds,
                    freshness_limit,
                )
        elif job.rt_key in failed_keys:
            status = "jvlink_unavailable"
        else:
            status = "no_odds_before_target"
        events.append({
            "rt_key": job.rt_key, "race_id": job.race_id, "snapshot_label": job.label,
            "target_datetime": job.target_datetime.isoformat(timespec="seconds"),
            "post_datetime": job.post_datetime.isoformat(timespec="seconds"),
            "acquired_at": acquired_at.isoformat(timespec="seconds"),
            "delay_seconds": round((acquired_at - job.target_datetime).total_seconds(), 3),
            "odds_dataspec": dataspec,
            "latest_happyo_datetime": (
                max(happyo_datetimes).isoformat(timespec="seconds")
                if happyo_datetimes else ""
            ),
            "source_age_seconds": (
                round(source_age_seconds, 3) if source_age_seconds is not None else None
            ),
            "freshness_limit_seconds": freshness_limit,
            "status": status,
            "rows_saved": rows,
        })
    paths = save_offset_snapshots(snapshots, output_dir, target_date)
    _save_events(events, output_dir, target_date)
    return paths, events


def run_scheduler(
    schedule: pd.DataFrame,
    target_date: dt.date,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sid: str = "KEIBA_AI",
    collection_delay_seconds: float = 2.0,
    include_past: bool = False,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    poll_lookback_seconds: float = DEFAULT_POLL_LOOKBACK_SECONDS,
    on_jobs_executed: Optional[JobsExecutedCallback] = None,
) -> bool:
    """全ジョブを時刻順に実行する。``Ctrl+C`` で安全に停止できる。

    ``on_jobs_executed`` を渡すと、ジョブ群を保存した直後に取得イベントの
    リストを渡して呼び出す（例外は記録して取得を継続する）。

    Returns:
        bool: 全ジョブを実行し終えたら True、``Ctrl+C`` で停止したら False
    """
    def notify(events: list[dict[str, object]]) -> None:
        if on_jobs_executed is None or not events:
            return
        try:
            on_jobs_executed(events)
        except Exception:
            logger.exception("ジョブ実行後コールバックでエラーが発生しました（取得は継続）")

    if collection_delay_seconds < 0:
        raise ValueError("collection_delay_seconds は 0 以上で指定してください")
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds は 0 より大きく指定してください")
    if poll_lookback_seconds < 0:
        raise ValueError("poll_lookback_seconds は 0 以上で指定してください")
    all_jobs = build_snapshot_jobs(schedule)
    now = _now_jst_naive()
    past_jobs = [job for job in all_jobs if job.target_datetime < now]
    jobs = [job for job in all_jobs if job.target_datetime >= now]
    skipped = past_jobs if not include_past else []
    if skipped:
        logger.warning("起動時点を過ぎた %d ジョブをスキップします（--include-past で取得可能）", len(skipped))
        _save_events([
            {
                "rt_key": job.rt_key, "race_id": job.race_id, "snapshot_label": job.label,
                "target_datetime": job.target_datetime.isoformat(timespec="seconds"),
                "post_datetime": job.post_datetime.isoformat(timespec="seconds"),
                "acquired_at": now.isoformat(timespec="seconds"),
                "delay_seconds": round((now - job.target_datetime).total_seconds(), 3),
                "odds_dataspec": "",
                "latest_happyo_datetime": "",
                "source_age_seconds": None,
                "freshness_limit_seconds": None,
                "status": "skipped_at_startup", "rows_saved": 0,
            } for job in skipped
        ], output_dir, target_date)
    if not jobs and not (include_past and past_jobs):
        logger.warning("実行対象のジョブがありません")
        return True

    logger.info(
        "時点別オッズスケジューラ開始: リアルタイム %d ジョブ、履歴復元 %d ジョブ",
        len(jobs),
        len(past_jobs) if include_past else 0,
    )
    jv = create_jvlink(sid)
    index = 0
    histories = _empty_histories()
    try:
        if include_past and past_jobs:
            paths, events = execute_jobs(
                jv,
                past_jobs,
                output_dir=output_dir,
                target_date=target_date,
                now=now,
                dataspec=DATASPEC_O1_HISTORY,
                check_freshness=False,
            )
            logger.info(
                "過去ジョブ復元完了: %d 件、保存先 %d ファイル",
                len(events),
                len(paths),
            )
            notify(events)

        next_poll_at = (
            max(
                _now_jst_naive(),
                jobs[0].target_datetime
                - dt.timedelta(seconds=poll_lookback_seconds),
            )
            if jobs else None
        )
        while index < len(jobs):
            now = _now_jst_naive()
            if next_poll_at is not None and now >= next_poll_at:
                poll_keys = _active_poll_keys(
                    jobs,
                    index,
                    now,
                    lookback_seconds=poll_lookback_seconds,
                    collection_delay_seconds=collection_delay_seconds,
                )
                if poll_keys:
                    failed = _fetch_into_histories(
                        jv,
                        poll_keys,
                        histories,
                        acquired_at=now,
                        dataspec=DATASPEC_O1_CURRENT,
                    )
                    logger.debug(
                        "速報オッズ先行取得: %d レース、失敗 %d レース",
                        len(poll_keys),
                        len(failed),
                    )
                    next_poll_at = now + dt.timedelta(seconds=poll_interval_seconds)
                else:
                    next_poll_at = max(
                        now + dt.timedelta(seconds=poll_interval_seconds),
                        jobs[index].target_datetime
                        - dt.timedelta(seconds=poll_lookback_seconds),
                    )

            next_time = jobs[index].target_datetime + dt.timedelta(
                seconds=collection_delay_seconds
            )
            wake_times = [next_time]
            if next_poll_at is not None:
                wake_times.append(next_poll_at)
            wait_seconds = (min(wake_times) - _now_jst_naive()).total_seconds()
            if wait_seconds > 0:
                time.sleep(min(wait_seconds, 30.0))
                continue
            now = _now_jst_naive()
            due: list[SnapshotJob] = []
            while index < len(jobs):
                due_time = jobs[index].target_datetime + dt.timedelta(seconds=collection_delay_seconds)
                if due_time > now:
                    break
                due.append(jobs[index])
                index += 1
            paths, events = execute_jobs(
                jv,
                due,
                output_dir=output_dir,
                target_date=target_date,
                now=now,
                dataspec=DATASPEC_O1_CURRENT,
                histories=histories,
                check_freshness=True,
            )
            logger.info("ジョブ実行完了: %d 件、保存先 %d ファイル", len(events), len(paths))
            notify(events)
    except KeyboardInterrupt:
        logger.info("時点別オッズスケジューラを停止しました")
        return False
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JRA当日の発走時刻に連動してレース別の時点別オッズCSVを作成する")
    parser.add_argument("--headed", action="store_true", help="出馬表取得用ブラウザを表示する")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sid", default="KEIBA_AI", help="JVInit のソフトウェアID")
    parser.add_argument("--collection-delay-seconds", type=float, default=2.0,
                        help="基準時刻後にJV-Linkを読むまでの待機秒数（既定: 2）")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help="各基準時刻前の速報オッズ先行取得間隔（既定: 30秒）",
    )
    parser.add_argument(
        "--poll-lookback-seconds",
        type=float,
        default=DEFAULT_POLL_LOOKBACK_SECONDS,
        help="各基準時刻の何秒前から先行取得するか（既定: 120秒）",
    )
    parser.add_argument("--include-past", action="store_true", help="過ぎた基準時刻も履歴から復元を試みる")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = _parse_args()
    target_date = dt.datetime.now(JST).date()
    schedule = get_today_schedule(target_date, headless=not args.headed)
    run_scheduler(
        schedule, target_date, output_dir=args.output_dir, sid=args.sid,
        collection_delay_seconds=args.collection_delay_seconds, include_past=args.include_past,
        poll_interval_seconds=args.poll_interval_seconds,
        poll_lookback_seconds=args.poll_lookback_seconds,
    )


if __name__ == "__main__":
    main()
