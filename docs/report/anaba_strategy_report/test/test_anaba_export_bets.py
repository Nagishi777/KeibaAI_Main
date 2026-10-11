"""ベット明細 CSV の生成ロジックのテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.anaba.export_bets import (
    split_by_method,
    summarize,
    summarize_method,
)


def _detail() -> pd.DataFrame:
    """build() が返す形に合わせた最小の明細表。"""
    return pd.DataFrame(
        {
            "date": ["2024-01-06", "2024-01-06", "2024-02-03"],
            "race_id": ["202408010101", "202408010102", "202408010201"],
            "won": [0, 1, 0],
            "odds_1m": [70.6, 21.9, 30.0],
            "stake_flat": [1000.0, 1000.0, 1000.0],
            "return_flat": [0.0, 21900.0, 0.0],
            "pnl_flat": [-1000.0, 20900.0, -1000.0],
            "bankroll_flat": [299000.0, 319900.0, 318900.0],
            "bankroll_kelly": [299900.0, 320000.0, 319000.0],
        }
    )


def test_summarize_groups_by_month() -> None:
    g = summarize(_detail())
    assert g["month"].tolist() == ["2024-01", "2024-02"]


def test_summarize_counts_bets_and_hits() -> None:
    g = summarize(_detail()).set_index("month")
    assert g.loc["2024-01", "bets"] == 2
    assert g.loc["2024-01", "hits"] == 1
    assert g.loc["2024-02", "hits"] == 0


def test_summarize_roi_matches_return_over_stake() -> None:
    g = summarize(_detail()).set_index("month")
    # 2024-01: 21,900 returned on 2,000 staked
    assert np.isclose(g.loc["2024-01", "roi_flat"], 21900.0 / 2000.0)
    assert np.isclose(g.loc["2024-02", "roi_flat"], 0.0)


def test_summarize_carries_month_end_bankroll() -> None:
    g = summarize(_detail()).set_index("month")
    # 月末の残高は、その月の最後の行の値になる
    assert g.loc["2024-01", "bankroll_flat_end"] == 319900.0
    assert g.loc["2024-02", "bankroll_flat_end"] == 318900.0
    assert g.loc["2024-01", "bankroll_kelly_end"] == 320000.0


def test_summarize_pnl_is_return_minus_stake() -> None:
    g = summarize(_detail()).set_index("month")
    for month in ("2024-01", "2024-02"):
        row = g.loc[month]
        assert np.isclose(row["pnl_flat"], row["return_flat"] - row["stake_flat"])


def _full_detail() -> pd.DataFrame:
    """split_by_method / summarize_method 用の、共通列を全て持つ明細表。"""
    base = {
        "date": ["2024-01-06", "2024-01-06", "2024-02-03"],
        "post_time": ["09:50", "10:20", "11:00"],
        "race_id": ["202408010101", "202408010102", "202408010201"],
        "race_no": ["01", "02", "01"],
        "racecourse": ["京都", "京都", "小倉"],
        "race_name": ["3歳未勝利", "3歳未勝利", "4歳以上1勝クラス"],
        "race_class": ["未勝利", "未勝利", "1勝クラス"],
        "distance": [1800, 1800, 1200],
        "track_type": ["ダート", "ダート", "芝"],
        "track_condition": ["良", "良", "良"],
        "field_size": [16, 16, 14],
        "horse_number": [11, 3, 7],
        "horse_name": ["馬A", "馬B", "馬C"],
        "jockey": ["騎手A", "騎手B", "騎手C"],
        "popularity": [13.0, 5.0, 8.0],
        "odds_5m": [82.2, 30.0, 45.0],
        "odds_1m": [70.6, 21.9, 30.0],
        "odds_drop_pct": [-14.11, -27.0, -33.33],
        "drift_5_1": [-0.1521, -0.315, -0.405],
        "inc_share_5_1": [0.013, 0.05, 0.03],
        "pool_1m_oku": [5.37, 6.0, 4.2],
        "p": [0.0174, 0.0973, 0.03],
        "implied_prob": [0.0113, 0.0365, 0.0267],
        "edge_ratio": [1.53, 2.66, 1.12],
        "ev": [1.23, 2.13, 0.9],
        "ev_rank": [1.0, 1.0, 1.0],
        "finish_position": [11, 1, 5],
        "hit": ["不的中", "的中", "不的中"],
        "won": [0, 1, 0],
        "final_odds": [70.6, 21.9, 30.0],
        "payout_win": [0.0, 2190.0, 0.0],
        "stake_flat": [1000.0, 1000.0, 1000.0],
        "return_flat": [0.0, 21900.0, 0.0],
        "pnl_flat": [-1000.0, 20900.0, -1000.0],
        "bankroll_flat": [299000.0, 319900.0, 318900.0],
        "stake_kelly": [100.0, 500.0, 300.0],
        "return_kelly": [0.0, 10950.0, 0.0],
        "pnl_kelly": [-100.0, 10450.0, -300.0],
        "bankroll_kelly": [299900.0, 310350.0, 310050.0],
    }
    return pd.DataFrame(base)


def test_split_by_method_flat_renames_columns_without_suffix() -> None:
    out = split_by_method(_full_detail(), "flat")
    assert list(out.columns[-4:]) == ["stake", "return", "pnl", "bankroll"]
    assert "stake_flat" not in out.columns
    assert "stake_kelly" not in out.columns


def test_split_by_method_flat_and_kelly_have_same_common_columns() -> None:
    flat = split_by_method(_full_detail(), "flat")
    kelly = split_by_method(_full_detail(), "kelly")
    common_flat = [
        c for c in flat.columns if c not in ("stake", "return", "pnl", "bankroll")
    ]
    common_kelly = [
        c for c in kelly.columns if c not in ("stake", "return", "pnl", "bankroll")
    ]
    assert common_flat == common_kelly


def test_split_by_method_preserves_values() -> None:
    kelly = split_by_method(_full_detail(), "kelly")
    assert kelly["stake"].tolist() == [100.0, 500.0, 300.0]
    assert kelly["bankroll"].tolist() == [299900.0, 310350.0, 310050.0]
    flat = split_by_method(_full_detail(), "flat")
    assert flat["stake"].tolist() == [1000.0, 1000.0, 1000.0]


def test_split_by_method_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="flat.*kelly"):
        split_by_method(_full_detail(), "martingale")


def test_summarize_method_matches_manual_roi() -> None:
    g = summarize_method(_full_detail(), "kelly").set_index("month")
    # 2024-01: stake 100+500=600, return 0+10950=10950
    assert np.isclose(g.loc["2024-01", "roi"], 10950.0 / 600.0)
    assert g.loc["2024-01", "hits"] == 1
    assert g.loc["2024-02", "hits"] == 0


def test_summarize_method_flat_matches_summarize_columns() -> None:
    d = _full_detail()
    combined = summarize(d).set_index("month")
    flat_only = summarize_method(d, "flat").set_index("month")
    assert np.isclose(
        combined.loc["2024-01", "roi_flat"], flat_only.loc["2024-01", "roi"]
    )
    assert (
        combined.loc["2024-01", "bankroll_flat_end"]
        == flat_only.loc["2024-01", "bankroll_end"]
    )
