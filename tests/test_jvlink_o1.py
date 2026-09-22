import unittest

from src.scraping._jvlink_o1 import fetch_odds_history, parse_o1_record


def _make_o1_record(*, include_record_separator: bool = True) -> str:
    """仕様どおりの最小O1レコードを組み立てる。"""
    data = [" "] * 960

    def put(start: int, value: str) -> None:
        data[start:start + len(value)] = value

    put(0, "O1")
    put(2, "1")
    put(3, "20260919")
    put(11, "2026")
    put(15, "0919")
    put(19, "06")
    put(21, "04")
    put(23, "07")
    put(25, "01")
    put(27, "09190940")
    put(35, "18")
    put(37, "18")
    put(39, "777")
    put(42, "3")

    # 単勝、複勝、枠連を各1件だけ設定する。
    put(43, "01002501")
    put(267, "010020003001")
    put(603, "120012301")

    # O1の票数合計は各11桁。
    put(927, "00000123456")
    put(938, "00000234567")
    put(949, "00000345678")

    record = "".join(data)
    return record + "\r\n" if include_record_separator else record


class ParseO1RecordTest(unittest.TestCase):
    def test_parses_official_962_byte_record(self) -> None:
        parsed = parse_o1_record(_make_o1_record())

        self.assertEqual(parsed["tansho"][0]["odds_win"], 2.5)
        self.assertEqual(parsed["tansho"][0]["hyosu_total"], 123456)
        self.assertEqual(parsed["fukusho"][0]["odds_place_min"], 2.0)
        self.assertEqual(parsed["fukusho"][0]["odds_place_max"], 3.0)
        self.assertEqual(parsed["fukusho"][0]["hyosu_total"], 234567)
        self.assertEqual(parsed["wakuren"][0]["odds_wakuren"], 12.3)
        self.assertEqual(parsed["wakuren"][0]["hyosu_total"], 345678)

    def test_accepts_record_without_crlf(self) -> None:
        parsed = parse_o1_record(_make_o1_record(include_record_separator=False))
        self.assertEqual(parsed["tansho"][0]["race_id"], "202606040701")

    def test_rejects_truncated_data_part(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected at least 960"):
            parse_o1_record(_make_o1_record(include_record_separator=False)[:-1])


class _UnavailableJVLink:
    def __init__(self) -> None:
        self.open_calls: list[tuple[str, str]] = []

    def JVRTOpen(self, dataspec: str, key: str) -> int:
        self.open_calls.append((dataspec, key))
        return -1


class FetchOddsHistoryTest(unittest.TestCase):
    def test_passes_requested_dataspec_to_jvrtopen(self) -> None:
        jv = _UnavailableJVLink()

        _, failed = fetch_odds_history(
            jv, ["202609190601"], dataspec="0B31"
        )

        self.assertEqual(jv.open_calls, [("0B31", "202609190601")])
        self.assertEqual(failed, ["202609190601"])


if __name__ == "__main__":
    unittest.main()
