"""JV-Link の速報オッズ O1 レコードを読むための共通処理。"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Iterable, Iterator, Optional, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger(__name__)
JST = ZoneInfo("Asia/Tokyo")

DATASPEC_O1_CURRENT = "0B31"
DATASPEC_O1_HISTORY = "0B41"
# 後方互換用。時系列履歴を取得する既存コードでは従来どおり 0B41 を使う。
DATASPEC_O1 = DATASPEC_O1_HISTORY
BET_TYPES = ("tansho", "fukusho", "wakuren")
KEY_COLUMN = {
    "tansho": "umaban",
    "fukusho": "umaban",
    "wakuren": "kumiban",
}
JVREAD_BUFFER_SIZE = 200_000
# JV-Data仕様のO1レコードは、データ部960バイト + CR/LF 2バイト。
# JVReadの呼び出し方法によっては末尾のCR/LFが含まれないことがあるため、
# 解析に必要なデータ部の長さを下限として検証する。
O1_RECORD_LENGTH = 962
O1_DATA_LENGTH = 960


class JVLink(Protocol):
    """このモジュールが利用する JV-Link COM オブジェクトの最小インターフェース。"""

    def JVInit(self, sid: str) -> int: ...

    def JVRTOpen(self, dataspec: str, key: str) -> int: ...

    def JVRead(self, buff: str, size: int, filename: str) -> tuple[int, str, str]: ...

    def JVClose(self) -> int: ...


def create_jvlink(sid: str) -> JVLink:
    """JV-Link COM を初期化して返す。"""
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "pywin32 が必要です。Windows 環境で 'pip install pywin32' を実行してください。"
        ) from exc

    jv: JVLink = win32com.client.Dispatch("JVDTLab.JVLink")
    result = jv.JVInit(sid)
    if result != 0:
        raise RuntimeError(f"JVInit failed: {result}")
    return jv


def _to_int(value: str) -> int | None:
    return int(value) if value.isdigit() else None


def _to_odds(value: str) -> float | None:
    if not value.isdigit():
        return None
    return int(value) / 10


def _is_empty_slot(value: str) -> bool:
    return value.strip() in {"", "0", "00"}


def _happyo_datetime(year: str, value: str) -> dt.datetime:
    return dt.datetime(
        int(year), int(value[0:2]), int(value[2:4]), int(value[4:6]), int(value[6:8])
    )


def parse_o1_record(data: str) -> dict[str, list[dict[str, Any]]]:
    """O1（単勝・複勝・枠連時系列オッズ）を券種別の行に展開する。

    O1 は固定長レコードで、オッズ値は10倍された整数として格納されている。
    """
    if len(data) < O1_DATA_LENGTH:
        raise ValueError(
            f"O1 record is too short: {len(data)} characters "
            f"(expected at least {O1_DATA_LENGTH}; "
            f"official record length is {O1_RECORD_LENGTH} including CR/LF)"
        )

    year = data[11:15]
    venue_code = data[19:21]
    kai = data[21:23]
    day = data[23:25]
    race_number = data[25:27]
    header: dict[str, Any] = {
        "race_id": f"{year}{venue_code}{kai}{day}{race_number}",
        "data_kubun": data[2:3],
        "made_date": dt.date(int(data[3:7]), int(data[7:9]), int(data[9:11])),
        "keibajo_code": venue_code,
        "race_num": race_number,
        "happyo_datetime": _happyo_datetime(year, data[27:35]),
        "toroku_tosu": _to_int(data[35:37]),
        "syusso_tosu": _to_int(data[37:39]),
    }
    flags = {
        "tansho": data[39:40],
        "fukusho": data[40:41],
        "wakuren": data[41:42],
    }
    totals = {
        "tansho": _to_int(data[927:938]),
        "fukusho": _to_int(data[938:949]),
        "wakuren": _to_int(data[949:960]),
    }
    result: dict[str, list[dict[str, Any]]] = {bet_type: [] for bet_type in BET_TYPES}

    for index in range(0, 224, 8):
        chunk = data[43 + index: 43 + index + 8]
        umaban = chunk[:2]
        if not _is_empty_slot(umaban):
            result["tansho"].append({
                **header, "hatsubai_flag": flags["tansho"],
                "hyosu_total": totals["tansho"], "umaban": umaban,
                "odds_win": _to_odds(chunk[2:6]), "ninkijun": _to_int(chunk[6:8]),
            })

    for index in range(0, 336, 12):
        chunk = data[267 + index: 267 + index + 12]
        umaban = chunk[:2]
        if not _is_empty_slot(umaban):
            result["fukusho"].append({
                **header, "hatsubai_flag": flags["fukusho"],
                "hyosu_total": totals["fukusho"], "umaban": umaban,
                "odds_place_min": _to_odds(chunk[2:6]),
                "odds_place_max": _to_odds(chunk[6:10]),
                "ninkijun": _to_int(chunk[10:12]),
            })

    for index in range(0, 324, 9):
        chunk = data[603 + index: 603 + index + 9]
        kumiban = chunk[:2]
        if not _is_empty_slot(kumiban):
            result["wakuren"].append({
                **header, "hatsubai_flag": flags["wakuren"],
                "hyosu_total": totals["wakuren"], "kumiban": kumiban,
                "odds_wakuren": _to_odds(chunk[2:7]), "ninkijun": _to_int(chunk[7:9]),
            })
    return result


def iter_records(jv: JVLink) -> Iterator[str]:
    """Open 済みの JV-Link からすべてのレコードを返す。"""
    while True:
        result = jv.JVRead(" " * JVREAD_BUFFER_SIZE, JVREAD_BUFFER_SIZE, "")
        code = result[0]
        if code > 0:
            yield result[1]
        elif code == 0:
            return
        elif code == -1:
            # 次のファイルを準備中。JV-Link の仕様に従い再試行する。
            continue
        else:
            raise RuntimeError(f"JVRead failed: {code}")


def collect_o1_rows(jv: JVLink) -> dict[str, list[dict[str, Any]]]:
    """Open 済みの JV-Link から O1 レコードだけを券種別に収集する。"""
    result: dict[str, list[dict[str, Any]]] = {bet_type: [] for bet_type in BET_TYPES}
    for data in iter_records(jv):
        if data[:2] != "O1":
            continue
        try:
            parsed = parse_o1_record(data)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "不正なO1レコードをスキップします (length=%d): %s", len(data), exc
            )
            continue
        for bet_type in BET_TYPES:
            result[bet_type].extend(parsed[bet_type])
    return result


def fetch_odds_history(
    jv: JVLink,
    rt_keys: Iterable[str],
    *,
    acquired_at: Optional[dt.datetime] = None,
    dataspec: str = DATASPEC_O1_HISTORY,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """複数レースの O1 レコードを取得する。

    ``0B41`` では発表履歴、``0B31`` では取得時点の速報値を返す。最新値だけに
    集約せず返すため、呼び出し側で複数回の取得結果を蓄積できる。
    """
    all_rows: dict[str, list[dict[str, Any]]] = {
        bet_type: [] for bet_type in BET_TYPES
    }
    failed: list[str] = []
    acquired_at = acquired_at or dt.datetime.now(JST)

    for rt_key in rt_keys:
        result = jv.JVRTOpen(dataspec, rt_key)
        if result != 0:
            logger.debug(
                "JVRTOpen failed: %s (dataspec=%s, rt_key=%s)",
                result,
                dataspec,
                rt_key,
            )
            failed.append(rt_key)
            continue

        try:
            parsed = collect_o1_rows(jv)
        finally:
            jv.JVClose()

        if not any(parsed.values()):
            logger.debug("オッズ未取得 (rt_key=%s)", rt_key)
            failed.append(rt_key)
            continue

        acquired_text = acquired_at.isoformat(timespec="seconds")
        for bet_type in BET_TYPES:
            all_rows[bet_type].extend(
                {
                    **row,
                    "rt_key": rt_key,
                    "acquired_at": acquired_text,
                }
                for row in parsed[bet_type]
            )

    return {
        bet_type: pd.DataFrame(rows)
        for bet_type, rows in all_rows.items()
    }, failed
