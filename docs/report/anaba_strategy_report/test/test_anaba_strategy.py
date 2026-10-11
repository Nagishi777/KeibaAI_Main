"""穴馬戦略の選別ルールと賭け金計算のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.anaba.strategy import AnabaRule, kelly_stake, select


def _frame() -> pd.DataFrame:
    """1レース分のダミー予測。drift は odds_1m/odds_5m から決まる。"""
    return pd.DataFrame(
        {
            "race_id": ["R1"] * 6,
            "horse_number": [1, 2, 3, 4, 5, 6],
            "p": [0.30, 0.060, 0.050, 0.040, 0.030, 0.0020],
            # 1:人気薄でない / 2,3,4:条件を満たす / 5:オッズ上昇 / 6:高すぎる
            "odds_1m": [2.0, 25.0, 30.0, 40.0, 50.0, 400.0],
            "odds_5m": [2.0, 30.0, 36.0, 50.0, 40.0, 500.0],
        }
    )


def test_select_keeps_only_falling_odds_in_band() -> None:
    picked = select(_frame())
    # 馬1: オッズ2.0 は odds_min 未満で除外
    # 馬5: オッズが上がっている（drift>0）ので除外
    # 馬6: 400倍は odds_max 超で除外
    assert sorted(picked["horse_number"].tolist()) == [2, 3, 4]
    assert (picked["ev"] >= 1.2).all()
    assert (picked["drift_5_1"] <= -0.05).all()


def test_select_respects_max_bets_per_race() -> None:
    rule = AnabaRule(max_bets_per_race=2)
    picked = select(_frame(), rule)
    assert len(picked) == 2
    # 期待値の高い順に採用される
    assert picked["ev"].is_monotonic_decreasing


def test_select_computes_drift_when_absent_and_reuses_when_present() -> None:
    df = _frame()
    assert "drift_5_1" not in df.columns
    assert len(select(df)) == 3

    # 明示的に与えた drift が尊重される（全馬を上昇扱いにすると0点）
    df2 = df.copy()
    df2["drift_5_1"] = 0.5
    assert len(select(df2)) == 0


def test_select_raises_on_missing_columns() -> None:
    with pytest.raises(ValueError, match="必要な列がありません"):
        select(_frame().drop(columns=["odds_5m"]))


def test_select_raises_on_nan_instead_of_imputing() -> None:
    df = _frame()
    df.loc[1, "p"] = np.nan
    with pytest.raises(ValueError, match="欠損"):
        select(df)


def test_kelly_stake_never_silently_drops_a_positive_edge_bet() -> None:
    p = np.array([0.06, 0.05, 0.04])
    odds = np.array([25.0, 30.0, 40.0])
    stake = kelly_stake(p, odds, 100_000.0)
    assert (stake >= 100.0).all()


def test_kelly_stake_zero_for_negative_edge() -> None:
    # p*odds < 1 なので賭けない
    stake = kelly_stake(np.array([0.01]), np.array([20.0]), 100_000.0)
    assert stake[0] == 0.0


def test_kelly_stake_caps_each_bet_by_bankroll_fraction() -> None:
    stake = kelly_stake(
        np.array([0.5]), np.array([10.0]), 100_000.0, max_fraction_per_bet=0.02
    )
    assert stake[0] <= 100_000.0 * 0.02


def test_kelly_stake_scales_with_bankroll() -> None:
    p, odds = np.array([0.05]), np.array([30.0])
    small = kelly_stake(p, odds, 200_000.0)[0]
    large = kelly_stake(p, odds, 2_000_000.0)[0]
    assert large > small


def test_kelly_stake_rejects_bankroll_too_small_for_unit() -> None:
    with pytest.raises(ValueError, match="購入単位"):
        kelly_stake(np.array([0.05]), np.array([30.0]), 3_000.0)
