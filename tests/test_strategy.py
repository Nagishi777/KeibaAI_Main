"""単勝モデル戦略の購入判定ルールのテスト。"""
import unittest

import numpy as np
import pandas as pd

from src.simulator.features import DRIFT_COL, ODDS_COL, POOL_COL
from src.simulator.strategy import (
    REASON_BET,
    REASON_DRIFT,
    REASON_EV,
    REASON_MAX_BETS,
    REASON_ODDS_HIGH,
    REASON_ODDS_LOW,
    REASON_POOL,
    REASON_SEP,
    WinRule,
    apply_rule,
    rule_from_config,
)

# 条件を全て満たす基準の1頭（EV = 0.10 * 20 = 2.00）
PASSING = {
    ODDS_COL: 20.0,
    DRIFT_COL: -0.30,
    POOL_COL: 10e8,
}


def _frame(rows: list[dict]) -> pd.DataFrame:
    """race_id / horse_number を補って判定用の表を作る。"""
    out = []
    for i, row in enumerate(rows):
        merged = dict(PASSING)
        merged.update(row)
        merged.setdefault('race_id', '1')
        merged.setdefault('horse_number', i + 1)
        out.append(merged)
    return pd.DataFrame(out)


def _reasons(frame: pd.DataFrame, index: int) -> set[str]:
    value = frame.loc[index, 'skip_reason']
    return set(value.split(REASON_SEP)) if value else set()


class ApplyRuleTest(unittest.TestCase):
    def test_passing_horse_is_bought_with_flat_stake(self) -> None:
        frame = _frame([{}])
        out = apply_rule(frame, np.array([0.10]))
        self.assertEqual(out.loc[0, 'skip_reason'], REASON_BET)
        self.assertAlmostEqual(out.loc[0, 'ev'], 2.0)

    def test_each_condition_rejects_with_its_own_reason(self) -> None:
        cases = [
            ({ODDS_COL: 5.0}, 0.30, REASON_ODDS_LOW),       # オッズ下限未満
            ({ODDS_COL: 200.0}, 0.10, REASON_ODDS_HIGH),    # オッズ上限超
            ({DRIFT_COL: +0.10}, 0.10, REASON_DRIFT),       # オッズが上がった
            ({POOL_COL: 1e8}, 0.10, REASON_POOL),           # 総投票額 1億円
            ({}, 0.01, REASON_EV),                          # EV = 0.2
        ]
        for overrides, proba, expected in cases:
            with self.subTest(expected=expected):
                out = apply_rule(_frame([overrides]), np.array([proba]))
                self.assertIn(expected, _reasons(out, 0))
                self.assertNotEqual(out.loc[0, 'skip_reason'], REASON_BET)

    def test_drift_boundary_is_inclusive(self) -> None:
        """ちょうど -0.05 は「5%以上下がった」として採用する。"""
        out = apply_rule(_frame([{DRIFT_COL: -0.05}]), np.array([0.10]))
        self.assertEqual(out.loc[0, 'skip_reason'], REASON_BET)

    def test_ev_and_pool_boundaries_are_inclusive(self) -> None:
        # EV ちょうど 1.05（= 0.0525 * 20）と総投票額ちょうど 5億円
        out = apply_rule(
            _frame([{POOL_COL: 5e8}]), np.array([1.05 / 20.0])
        )
        self.assertEqual(out.loc[0, 'skip_reason'], REASON_BET)

    def test_multiple_failures_are_all_recorded(self) -> None:
        """複数の理由で落ちた馬は理由を連結して残す。"""
        out = apply_rule(
            _frame([{ODDS_COL: 5.0, POOL_COL: 1e8, DRIFT_COL: +0.2}]),
            np.array([0.01]),
        )
        self.assertEqual(
            _reasons(out, 0),
            {REASON_ODDS_LOW, REASON_POOL, REASON_DRIFT, REASON_EV},
        )

    def test_only_highest_ev_horse_per_race_is_bought(self) -> None:
        """同一レースで条件を満たす馬が複数いても購入は1頭だけ。"""
        frame = _frame([
            {ODDS_COL: 20.0, 'horse_number': 1},
            {ODDS_COL: 50.0, 'horse_number': 2},   # EV が最大
            {ODDS_COL: 30.0, 'horse_number': 3},
        ])
        out = apply_rule(frame, np.array([0.10, 0.10, 0.10]))
        bets = out[out['skip_reason'] == REASON_BET]
        self.assertEqual(len(bets), 1)
        self.assertEqual(int(bets.iloc[0]['horse_number']), 2)
        # 残りは「レース内2位以下」として記録される
        others = out[out['horse_number'] != 2]
        self.assertTrue(
            all(REASON_MAX_BETS in _reasons(out, i) for i in others.index)
        )

    def test_separate_races_each_get_one_bet(self) -> None:
        """レースが複数あれば、それぞれで1頭ずつ買う。"""
        frame = _frame([
            {'race_id': '1', 'horse_number': 1},
            {'race_id': '2', 'horse_number': 1},
        ])
        out = apply_rule(frame, np.array([0.10, 0.10]))
        bets = out[out['skip_reason'] == REASON_BET]
        self.assertEqual(sorted(bets['race_id']), ['1', '2'])
        self.assertEqual(len(bets), 2)

    def test_missing_values_stop_processing(self) -> None:
        """判断に使う値に欠損があれば補完せず停止する。"""
        with self.assertRaises(ValueError):
            apply_rule(_frame([{DRIFT_COL: np.nan}]), np.array([0.10]))

    def test_missing_columns_stop_processing(self) -> None:
        frame = _frame([{}]).drop(columns=[POOL_COL])
        with self.assertRaises(ValueError):
            apply_rule(frame, np.array([0.10]))


class PoolPercentileTest(unittest.TestCase):
    """分位点モード: 絶対額ではなく「その日の中での順位」で絞る。

    当日データ（速報0B31）は学習データ（履歴0B41）より総票数が9〜13倍
    小さく、絶対額のしきい値が使えない。分位点ならスケール差に影響されない。
    """

    def _frame_with_pools(self, pools: list[float]) -> pd.DataFrame:
        """レースごとに総投票額が違う表を作る（他条件は全て通る）。"""
        return _frame([
            {'race_id': str(i), 'horse_number': 1, POOL_COL: pool}
            for i, pool in enumerate(pools)
        ])

    def test_percentile_mode_keeps_top_half(self) -> None:
        # 4レース・上位50% なので厚い2レース（分位点 0.75 / 1.00）だけ残る
        frame = self._frame_with_pools([1e8, 2e8, 3e8, 4e8])
        rule = WinRule(min_pool_pctile=0.50)
        out = apply_rule(frame, np.array([0.10] * 4), rule)
        kept = set(out[out['skip_reason'] == REASON_BET]['race_id'])
        self.assertEqual(kept, {'2', '3'})

    def test_percentile_keeps_the_intended_fraction(self) -> None:
        """上位X%の指定どおりの割合が残る（境界の取り方の確認）。"""
        pools = [float(i) * 1e8 for i in range(1, 21)]   # 20レース
        for pctile, expected in [(0.50, 10), (0.70, 6), (0.30, 14)]:
            with self.subTest(pctile=pctile):
                out = apply_rule(
                    self._frame_with_pools(pools), np.array([0.10] * 20),
                    WinRule(min_pool_pctile=pctile),
                )
                self.assertEqual(
                    len(out[out['skip_reason'] == REASON_BET]), expected
                )

    def test_percentile_mode_ignores_absolute_scale(self) -> None:
        """全体を1/10にしても同じレースが残る（スケール非依存）。"""
        rule = WinRule(min_pool_pctile=0.50)
        big = apply_rule(self._frame_with_pools([1e8, 2e8, 3e8, 4e8]),
                         np.array([0.10] * 4), rule)
        small = apply_rule(self._frame_with_pools([1e7, 2e7, 3e7, 4e7]),
                           np.array([0.10] * 4), rule)
        self.assertEqual(
            set(big[big['skip_reason'] == REASON_BET]['race_id']),
            set(small[small['skip_reason'] == REASON_BET]['race_id']),
        )
        # 絶対額モードなら小さい側は全滅する（対比）
        abs_rule = WinRule(min_pool_yen=5e8)
        out = apply_rule(self._frame_with_pools([1e7, 2e7, 3e7, 4e7]),
                         np.array([0.10] * 4), abs_rule)
        self.assertEqual(len(out[out['skip_reason'] == REASON_BET]), 0)

    def test_percentile_takes_precedence_over_absolute(self) -> None:
        """両方設定されていたら分位点が優先される。"""
        # 絶対額5億円なら全滅するが、分位点なら上位が残る
        frame = self._frame_with_pools([1e7, 2e7, 3e7, 4e7])
        rule = WinRule(min_pool_yen=5e8, min_pool_pctile=0.50)
        out = apply_rule(frame, np.array([0.10] * 4), rule)
        self.assertEqual(len(out[out['skip_reason'] == REASON_BET]), 2)

    def test_rejected_races_get_pool_reason(self) -> None:
        frame = self._frame_with_pools([1e8, 4e8])
        out = apply_rule(frame, np.array([0.10, 0.10]),
                         WinRule(min_pool_pctile=0.50))
        thin = out[out['race_id'] == '0']
        self.assertIn(REASON_POOL, _reasons(out, thin.index[0]))


class RuleFromConfigTest(unittest.TestCase):
    def test_defaults_match_the_agreed_conditions(self) -> None:
        rule, _ = rule_from_config({})
        self.assertEqual(rule.min_ev, 1.05)
        self.assertEqual(rule.odds_min, 10.0)
        self.assertEqual(rule.odds_max, 150.0)
        self.assertEqual(rule.max_drift, -0.05)
        self.assertEqual(rule.min_pool_yen, 5e8)
        self.assertEqual(rule.max_bets_per_race, 1)
        self.assertEqual(rule.stake_yen, 100.0)

    def test_percentile_can_be_set_from_config(self) -> None:
        rule, _ = rule_from_config({'strategy': {'min_pool_pctile': 0.5}})
        self.assertEqual(rule.min_pool_pctile, 0.5)

    def test_config_overrides_are_applied(self) -> None:
        rule, used = rule_from_config({
            'strategy': {'min_ev': 1.20, 'min_pool_yen': 700000000}
        })
        self.assertEqual(rule.min_ev, 1.20)
        self.assertEqual(rule.min_pool_yen, 7e8)
        # 指定しなかった値は既定のまま
        self.assertEqual(rule.odds_min, 10.0)
        self.assertEqual(used['min_ev'], 1.20)


if __name__ == '__main__':
    unittest.main()
