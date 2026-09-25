"""JRA の出馬表から当日レースの発走時刻とオッズ取得用キーを収集する。

JRA の「競馬メニュー > 出馬表」ページは CNAME を伴う JavaScript 遷移を
使用する。本モジュールは Playwright で出馬表の開催ページを開き、各レースの
出馬表から発走時刻を取得する。

出力には、リアルタイムオッズ取得で使用する JV-Link の ``rt_key``
（``YYYYMMDD + 場コード + レース番号``）も含める。

``realtime_odds.py`` が内部利用する補助モジュール。通常はこのファイルを
直接実行せず、公開CLIの ``python -m src.scraping.realtime_odds`` を使用する。

出力先（既定）::

    data/processed/schedules/YYYYMMDD_jra_today_schedule.csv

必要なパッケージ::

    pip install playwright beautifulsoup4 pandas
    playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

JST = ZoneInfo("Asia/Tokyo")
BASE_URL = "https://www.jra.go.jp"
THISWEEK_URL = f"{BASE_URL}/keiba/thisweek/"
ACCESS_D_URL = f"{BASE_URL}/JRADB/accessD.html"
DEFAULT_OUTPUT_DIR = Path("data/processed/schedules")

# pw01dde + 区分2桁 + 場コード2桁 + 年4桁 + 回2桁 + 日2桁 + R2桁 + 日付8桁
# URL 内では末尾の / が %2F にエンコードされる場合もある。
RACE_CNAME_PATTERN = re.compile(
    r"([ps]w01dde\d{2}\d{2}\d{4}\d{2}\d{2}\d{2}\d{8}(?:/|%2F)[A-Za-z0-9]+)",
    re.IGNORECASE,
)
RACE_CNAME_FIELDS = re.compile(
    r"^[ps]w01dde(?P<category>\d{2})(?P<venue_code>\d{2})"
    r"(?P<year>\d{4})(?P<kai>\d{2})(?P<day>\d{2})(?P<race_number>\d{2})"
    r"(?P<race_date>\d{8})/[A-Za-z0-9]+$",
    re.IGNORECASE,
)
ACCESS_D_ONCLICK_PATTERN = re.compile(
    r"doAction\S?\(['\"](?:https?://www\.jra\.go\.jp)?/JRADB/accessD\.html['\"]"
    r"\s*,\s*['\"]([^'\"]+)['\"]\)",
    re.IGNORECASE,
)
ACCESS_D_URL_PATTERN = re.compile(
    r"accessD\.html[?&](?:amp;)?CNAME=([^\"'&\s]+)", re.IGNORECASE
)
POST_TIME_PATTERN = re.compile(
    r"発走時刻\s*[：:]\s*(?P<hour>\d{1,2})時\s*(?P<minute>\d{1,2})分"
)


class JRATodayScheduleScraperError(RuntimeError):
    """JRA 出馬表スクレイピングの失敗。"""


def current_jst_date() -> dt.date:
    """現在の日本時間の日付を返す。"""
    return dt.datetime.now(JST).date()


def parse_race_cnames(html: str, target_date: str) -> list[dict[str, str | int]]:
    """出馬表 HTML から対象日のレース別 CNAME と識別子を抽出する。

    Args:
        html: JRA 出馬表または開催レース一覧ページの HTML。
        target_date: 対象日（``YYYYMMDD``）。

    Returns:
        場・回・日・レース番号で一意にしたレース情報一覧。HTML に対象日の
        レースリンクが無い場合は空リスト。
    """
    races: dict[tuple[str, str, str, str, int], dict[str, str | int]] = {}
    for matched in RACE_CNAME_PATTERN.findall(html):
        cname = unquote(matched)
        # sw01dde はスマートフォン用。PC版ページを巡回するため pw のみを使う。
        if not cname.lower().startswith("pw01dde"):
            continue
        fields = RACE_CNAME_FIELDS.fullmatch(cname)
        if not fields or fields["race_date"] != target_date:
            continue

        race_number = int(fields["race_number"])
        if not 1 <= race_number <= 12:
            continue
        key = (
            fields["venue_code"], fields["year"], fields["kai"],
            fields["day"], race_number,
        )
        races.setdefault(
            key,
            {
                "access_d_cname": cname,
                "venue_code": fields["venue_code"],
                "year": fields["year"],
                "kai": fields["kai"],
                "day": fields["day"],
                "race_number": race_number,
                "race_date": fields["race_date"],
            },
        )
    return sorted(
        races.values(),
        key=lambda item: (
            str(item["venue_code"]), int(item["race_number"]),
        ),
    )


def parse_meeting_cnames(
    html: str, target_date: str
) -> list[dict[str, str | int]]:
    """ページ内の開催場切替リンクから、当日の全開催場を抽出する。

    ``parse_race_cnames`` はHTML内のすべてのCNAMEをレースとして扱うため、
    開催場切替リンクの末尾に含まれる「レース数」をレース番号と誤認し得る。
    この関数は ``accessD.html`` へ遷移するアンカーだけを対象にする。
    """
    meetings: dict[tuple[str, str, str, str], dict[str, str | int]] = {}
    soup = BeautifulSoup(html, "html.parser")

    for link in soup.find_all("a"):
        candidates: list[str] = []
        onclick = str(link.get("onclick", ""))
        onclick_match = ACCESS_D_ONCLICK_PATTERN.search(onclick)
        if onclick_match:
            candidates.append(onclick_match.group(1))

        for attribute in ("href", "data-app-link"):
            value = str(link.get(attribute, ""))
            url_match = ACCESS_D_URL_PATTERN.search(value)
            if url_match:
                candidates.append(url_match.group(1))

        for candidate in candidates:
            cname = unquote(candidate)
            if not cname.lower().startswith("pw01dde"):
                continue
            fields = RACE_CNAME_FIELDS.fullmatch(cname)
            if not fields or fields["race_date"] != target_date:
                continue
            key = (
                fields["venue_code"], fields["year"], fields["kai"], fields["day"]
            )
            meetings.setdefault(
                key,
                {
                    "access_d_cname": cname,
                    "venue_code": fields["venue_code"],
                    "year": fields["year"],
                    "kai": fields["kai"],
                    "day": fields["day"],
                    "race_date": fields["race_date"],
                },
            )

    return sorted(meetings.values(), key=lambda item: str(item["venue_code"]))


def parse_post_time(html: str) -> Optional[dt.time]:
    """出馬表 HTML に表示される ``発走時刻：HH時MM分`` を読み取る。"""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    matched = POST_TIME_PATTERN.search(text)
    if not matched:
        return None
    try:
        return dt.time(int(matched["hour"]), int(matched["minute"]))
    except ValueError:
        return None


def parse_race_name(html: str) -> str:
    """出馬表のレース名を取得する。取得不能時は空文字列を返す。"""
    soup = BeautifulSoup(html, "html.parser")
    heading = soup.find("h2")
    return heading.get_text(" ", strip=True) if heading else ""


def build_schedule_record(
    race: dict[str, str | int], post_time: dt.time, race_name: str = ""
) -> dict[str, str | int]:
    """CNAME の情報と発走時刻から保存用の1レースレコードを作る。"""
    race_date = str(race["race_date"])
    year = str(race["year"])
    venue_code = str(race["venue_code"])
    kai = str(race["kai"])
    day = str(race["day"])
    race_number = int(race["race_number"])
    date_value = dt.datetime.strptime(race_date, "%Y%m%d").date()
    post_datetime = dt.datetime.combine(date_value, post_time)

    return {
        "date": date_value.isoformat(),
        "time": post_time.strftime("%H:%M"),
        "post_datetime": post_datetime.isoformat(timespec="minutes"),
        "race_id": f"{year}{venue_code}{kai}{day}{race_number:02d}",
        "rt_key": f"{race_date}{venue_code}{race_number:02d}",
        "venue_code": venue_code,
        "kai": kai,
        "day": day,
        "race_number": race_number,
        "race_name": race_name,
        "access_d_cname": str(race["access_d_cname"]),
        "shutuba_url": f"{ACCESS_D_URL}?CNAME={quote(str(race['access_d_cname']), safe='')}",
    }


async def _get_page_html(page, url: str, timeout_ms: int) -> str:
    await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
    return await page.content()


async def _collect_schedule(
    target_date: dt.date,
    *,
    timeout_ms: int,
    rate_limit_seconds: float,
    headless: bool,
) -> list[dict[str, str | int]]:
    """Playwright で出馬表を巡回し、当日レースのレコードを作る。"""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise JRATodayScheduleScraperError(
            "playwright がありません。'pip install playwright' と "
            "'playwright install chromium' を実行してください。"
        ) from exc

    target_text = target_date.strftime("%Y%m%d")
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=headless)
        context = await browser.new_context(locale="ja-JP")
        try:
            page = await context.new_page()
            thisweek_html = await _get_page_html(page, THISWEEK_URL, timeout_ms)
            entry_meetings = parse_meeting_cnames(thisweek_html, target_text)
            if not entry_meetings:
                # HTML構造変更時のフォールバックとしてCNAMEの直接走査を使う。
                entry_races = parse_race_cnames(thisweek_html, target_text)
                entry_meetings = [
                    {
                        "access_d_cname": race["access_d_cname"],
                        "venue_code": race["venue_code"],
                        "year": race["year"],
                        "kai": race["kai"],
                        "day": race["day"],
                        "race_date": race["race_date"],
                    }
                    for race in entry_races
                ]
            if not entry_meetings:
                logger.warning("対象日の出馬表リンクが見つかりません: %s", target_date)
                return []

            meeting_map = {
                (
                    str(meeting["venue_code"]), str(meeting["year"]),
                    str(meeting["kai"]), str(meeting["day"]),
                ): meeting
                for meeting in entry_meetings
            }

            # 最初の開催ページには、同日に行われる他開催場への切替リンクがある。
            # ここを解析して thisweek ページに直接現れなかった開催場も追加する。
            first_meeting = entry_meetings[0]
            first_url = (
                f"{ACCESS_D_URL}?CNAME="
                f"{quote(str(first_meeting['access_d_cname']), safe='')}"
            )
            first_html = await _get_page_html(page, first_url, timeout_ms)
            for meeting in parse_meeting_cnames(first_html, target_text):
                key = (
                    str(meeting["venue_code"]), str(meeting["year"]),
                    str(meeting["kai"]), str(meeting["day"]),
                )
                meeting_map.setdefault(key, meeting)

            logger.info(
                "対象日の開催場: %s",
                ", ".join(sorted(str(item["venue_code"]) for item in meeting_map.values())),
            )

            race_map: dict[str, dict[str, str | int]] = {}
            first_key = (
                str(first_meeting["venue_code"]), str(first_meeting["year"]),
                str(first_meeting["kai"]), str(first_meeting["day"]),
            )
            for meeting_key, meeting in meeting_map.items():
                if meeting_key == first_key:
                    meeting_html = first_html
                else:
                    url = (
                        f"{ACCESS_D_URL}?CNAME="
                        f"{quote(str(meeting['access_d_cname']), safe='')}"
                    )
                    meeting_html = await _get_page_html(page, url, timeout_ms)

                venue_races = [
                    race for race in parse_race_cnames(meeting_html, target_text)
                    if str(race["venue_code"]) == str(meeting["venue_code"])
                    and str(race["year"]) == str(meeting["year"])
                    and str(race["kai"]) == str(meeting["kai"])
                    and str(race["day"]) == str(meeting["day"])
                ]
                logger.info(
                    "開催場 %s: %d レースを検出",
                    meeting["venue_code"], len(venue_races),
                )
                for race in venue_races:
                    race_map[str(race["access_d_cname"])] = race
                await asyncio.sleep(rate_limit_seconds)

            records: list[dict[str, str | int]] = []
            for race in sorted(
                race_map.values(),
                key=lambda item: (str(item["venue_code"]), int(item["race_number"])),
            ):
                url = (
                    f"{ACCESS_D_URL}?CNAME="
                    f"{quote(str(race['access_d_cname']), safe='')}"
                )
                race_html = await _get_page_html(page, url, timeout_ms)
                post_time = parse_post_time(race_html)
                if post_time is None:
                    logger.warning(
                        "発走時刻を取得できません: venue=%s R%s",
                        race["venue_code"], race["race_number"],
                    )
                    continue
                records.append(
                    build_schedule_record(race, post_time, parse_race_name(race_html))
                )
                await asyncio.sleep(rate_limit_seconds)
            return records
        finally:
            await context.close()
            await browser.close()


def scrape_jra_today_schedule(
    target_date: Optional[dt.date] = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    timeout_ms: int = 30_000,
    rate_limit_seconds: float = 0.5,
    headless: bool = True,
) -> pd.DataFrame:
    """出馬表から当日のレース時刻・オッズ取得用キーを取得して CSV 保存する。

    Returns:
        ``date, time, post_datetime, race_id, rt_key`` を含む発走時刻順の
        DataFrame。開催がない場合は空の DataFrame を返す。
    """
    if timeout_ms <= 0:
        raise ValueError("timeout_ms は 0 より大きくしてください")
    if rate_limit_seconds < 0:
        raise ValueError("rate_limit_seconds は 0 以上で指定してください")

    target_date = target_date or current_jst_date()
    records = asyncio.run(
        _collect_schedule(
            target_date,
            timeout_ms=timeout_ms,
            rate_limit_seconds=rate_limit_seconds,
            headless=headless,
        )
    )
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame

    frame = frame.sort_values(
        ["post_datetime", "venue_code", "race_number"], kind="stable"
    ).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{target_date:%Y%m%d}_jra_today_schedule.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("JRA 当日レース日程を保存: %s (%d レース)", output_path, len(frame))
    return frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JRA 出馬表から当日レースの発走時刻と rt_key を取得する"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"CSV 保存先（既定: {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    parser.add_argument("--rate-limit-seconds", type=float, default=0.5)
    parser.add_argument("--headed", action="store_true", help="ブラウザを表示して実行する")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = _parse_args()
    target_date = current_jst_date()
    frame = scrape_jra_today_schedule(
        target_date,
        output_dir=args.output_dir,
        timeout_ms=args.timeout_ms,
        rate_limit_seconds=args.rate_limit_seconds,
        headless=not args.headed,
    )
    if frame.empty:
        print("対象日の出馬表または発走時刻を取得できませんでした")
        return
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
