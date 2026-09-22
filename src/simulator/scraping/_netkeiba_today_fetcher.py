"""
当日（未開催）レース向けに netkeiba.com から情報を取得するモジュール。

取得先:
    - 当日レース ID 一覧 → race.netkeiba.com の race_list_sub.html
    - 出馬表ページ        → data/raw/html_race/{race_id}_shutuba.html
"""
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

from src.scraping.netkeiba_session import build_netkeiba_session
from src.scraping.race_parser import parse_shutuba_race_info
from src.scraping.horse_parser import parse_shutuba_horses

logger = logging.getLogger(__name__)


class NetkeibaTodayFetcher:
    """当日（未開催）レースの情報を race.netkeiba.com から取得するクラス。"""

    RACE_NETKEIBA_URL = "https://race.netkeiba.com"

    def __init__(self, config: dict) -> None:
        """初期化。

        Args:
            config: 設定辞書
        """
        self.config = config
        self.session = build_netkeiba_session(config)

        raw_data_dir = Path(config.get('raw_data_dir', 'data/raw'))
        self.html_race_dir = raw_data_dir / 'html_race'
        self.html_race_dir.mkdir(parents=True, exist_ok=True)

    def fetch_race_ids_by_date(self, target_date: str) -> List[str]:
        """指定日の JRA レース ID リストを race.netkeiba.com から取得する。

        Args:
            target_date: 対象日（YYYY-MM-DD）

        Returns:
            List[str]: 当日の JRA レース ID リスト（12 桁）
        """
        date_str = target_date.replace('-', '')
        url = f"{self.RACE_NETKEIBA_URL}/top/race_list_sub.html?kaisai_date={date_str}"
        headers = {
            'Referer': f'{self.RACE_NETKEIBA_URL}/top/race_list.html?kaisai_date={date_str}',
            'Accept-Language': 'ja,en-US;q=0.9,en;q=0.8',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        }
        id_pattern = re.compile(r'race_id=(\d{12})|/race/(\d{12})/')

        try:
            response = self.session.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')

            race_ids: set[str] = set()
            for link in soup.find_all('a', href=True):
                m = id_pattern.search(link['href'])
                if m:
                    race_id = m.group(1) or m.group(2)
                    if len(race_id) == 12:
                        venue_code = int(race_id[4:6])
                        if 1 <= venue_code <= 10:
                            race_ids.add(race_id)

            result = sorted(race_ids)
            logger.info(f"{target_date}: {len(result)} 件の JRA レースを取得")

            if not result:
                logger.warning(
                    f"race_list_sub.html から 0 件。"
                    f"レスポンス長: {len(response.content)} bytes, "
                    f"ステータス: {response.status_code}, URL: {response.url}"
                )
                logger.info(f"レスポンス先頭 300 文字: {response.content[:300]}")

            return result

        except Exception as e:
            logger.error(f"当日レース一覧取得エラー ({target_date}): {e}")
            return []

    def fetch_shutuba_html(self, race_id: str, overwrite: bool = False) -> Optional[Path]:
        """出馬表ページ（race.netkeiba.com）の HTML を取得して html_race/ に保存する。

        未開催レースのフォールバック用。ファイル名は "{race_id}_shutuba.html"。

        Args:
            race_id: レース ID（12 桁）
            overwrite: True の場合、既存キャッシュを上書きする

        Returns:
            Optional[Path]: 保存した HTML ファイルのパス（失敗時は None）
        """
        html_file = self.html_race_dir / f"{race_id}_shutuba.html"
        if html_file.exists() and not overwrite:
            return html_file

        url = f"{self.RACE_NETKEIBA_URL}/race/shutuba.html?race_id={race_id}"
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            html_file.write_bytes(response.content)
            logger.debug(f"出馬表 HTML 保存: {html_file.name}")
            return html_file
        except Exception as e:
            logger.error(f"出馬表 HTML 取得エラー (race_id={race_id}): {e}")
            return None

    def fetch_today_race_data(self, race_id: str) -> Tuple[Optional[dict], list]:
        """当日（未開催）レースの出馬表を取得し、レース情報と出走馬データを返す。

        Args:
            race_id: レース ID（12 桁）

        Returns:
            Tuple[Optional[dict], list]: (レース情報辞書, 出走馬データのリスト)。
                取得・パース失敗時はそれぞれ None / 空リスト。
        """
        html_path = self.fetch_shutuba_html(race_id)
        if not html_path:
            return None, []

        race_info = parse_shutuba_race_info(html_path)
        horses = parse_shutuba_horses(html_path)
        return race_info, horses
