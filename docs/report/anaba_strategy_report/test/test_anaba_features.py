"""特徴量セットの定義に関するテスト。"""

from src.backtest.anaba.features import FEATURES, FEATURES_TOP8


def test_all29_has_29_columns() -> None:
    assert len(FEATURES) == 29


def test_top8_has_8_columns() -> None:
    assert len(FEATURES_TOP8) == 8


def test_no_duplicate_columns() -> None:
    assert len(set(FEATURES)) == len(FEATURES)
    assert len(set(FEATURES_TOP8)) == len(FEATURES_TOP8)


def test_top8_is_subset_of_all29() -> None:
    # 削減版は build() が作る列だけで構成されている必要がある
    assert set(FEATURES_TOP8) <= set(FEATURES)


def test_top8_keeps_the_core_signal() -> None:
    # 戦略の中核（直前4分の資金流入とオッズ水準）は必ず残す
    assert "inc_share_5_1" in FEATURES_TOP8
    assert "log_odds_1m" in FEATURES_TOP8


def test_top8_drops_the_perfectly_collinear_pair() -> None:
    # is_turf と is_dirt は相関 1.00 の完全重複なので削減版には入れない
    assert "is_turf" not in FEATURES_TOP8
    assert "is_dirt" not in FEATURES_TOP8
