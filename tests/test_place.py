"""複勝モデル・複勝の購入判定・単勝との同時実行のテスト。"""
from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataclasses import replace

from src.buying.audit import AuditLogger
from src.buying.browser.base import BrowserContractError
from src.buying.browser.playwright_client import _bet_type_label
from src.buying.domain import BetIntent, BetType, BuyingMode, Credentials
from src.buying.ledger import Ledger
from src.buying.service import PurchaseService
from src.buying.strategy import (
    PLACE_STRATEGY_ID,
    STRATEGY_ID,
    SimulatorStrategy,
    combine_results,
)
from src.simulator.features import DRIFT_COL, FEATURE_COLS
from src.simulator.place_model import PlaceModel, load_meta, place_label, train_place_model
from src.simulator.place_strategy import (
    PLACE_ODDS_COL,
    REASON_PLACE_ODDS_MISSING,
    PlaceRule,
    apply_place_rule,
    place_rule_from_config,
)
from src.simulator.realtime_loader import load_today_place_odds
from src.simulator.strategy import REASON_DRIFT, REASON_MAX_BETS, REASON_ODDS_HIGH, REASON_ODDS_LOW
from test_buying import FakeBrowser, evidence, make_settings, race

import main as main_module


class PlaceLabelTest(unittest.TestCase):
    def test_small_fields_pay_two_places(self) -> None:
        finish = pd.Series([1, 2, 3, 3, 4, 1, 2, 3])
        starters = pd.Series([8, 8, 8, 8, 8, 7, 7, 7])
        # 同着の3着は両方入着。7頭以下は3着が対象外
        self.assertEqual(place_label(finish, starters).tolist(), [1, 1, 1, 1, 0, 1, 1, 0])


class ApplyPlaceRuleTest(unittest.TestCase):
    @staticmethod
    def _frame(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{'race_id': r, 'horse_number': i + 1, PLACE_ODDS_COL: odds, DRIFT_COL: drift}
             for i, (r, odds, drift) in enumerate(rows)]
        )

    def test_band_drift_ev_and_top3(self) -> None:
        df = self._frame([
            ('R1', 8.0, -0.2),   # 買い（EV 1.6）
            ('R1', 5.9, -0.2),   # 下限未満
            ('R1', 20.5, -0.2),  # 上限超
            ('R1', 10.0, 0.0),   # ドリフト不足
            ('R1', 7.0, -0.2),   # 買い（EV 1.4）
            ('R1', 6.0, -0.2),   # EV 1.2 はレース内4位なので落ちる
            ('R1', 6.5, -0.2),   # 買い（EV 1.3・3位）
            ('R2', float('nan'), -0.2),  # 複勝オッズ無し
        ])
        proba = np.full(len(df), 0.2)
        out = apply_place_rule(df, proba, PlaceRule())
        reasons = dict(zip(out['horse_number'], out['skip_reason']))
        self.assertEqual(reasons[1], '')
        self.assertEqual(reasons[2], REASON_ODDS_LOW)
        self.assertEqual(reasons[3], REASON_ODDS_HIGH)
        self.assertEqual(reasons[4], REASON_DRIFT)
        self.assertEqual(reasons[5], '')
        self.assertEqual(reasons[7], '')            # EV 1.30 は3位
        self.assertEqual(reasons[6], REASON_MAX_BETS)  # EV 1.20 は4位
        self.assertEqual(reasons[8], REASON_PLACE_ODDS_MISSING)
        self.assertAlmostEqual(float(out.loc[out['horse_number'] == 1, 'ev'].iloc[0]), 1.6)

    def test_config_overrides(self) -> None:
        rule, _ = place_rule_from_config({'place_strategy': {'odds_max': 15}}, min_ev=1.2)
        self.assertEqual((rule.odds_min, rule.odds_max, rule.min_ev), (6.0, 15.0, 1.2))


class LoadTodayPlaceOddsTest(unittest.TestCase):
    def test_reads_requested_snapshot_per_race(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            folder = root / '20261011'
            folder.mkdir()
            rows = []
            for race_id in ('202605040401', '202605040402'):
                for label, low in (('2m', 6.0), ('3m', 9.0)):
                    rows.append({
                        'race_id': race_id, 'umaban': '01', 'snapshot_label': label,
                        'odds_place_min': low, 'odds_place_max': low * 2,
                        'happyo_datetime': '2026-10-11 10:00:00',
                    })
                pd.DataFrame([r for r in rows if r['race_id'] == race_id]).to_csv(
                    folder / f'{race_id}_fukusho_realtimeodds.csv', index=False,
                    encoding='utf-8-sig',
                )
            out = load_today_place_odds(
                dt.date(2026, 10, 11), root,
                {'202605040401': '2m', '202605040402': '3m'},
            )
        got = dict(zip(out['race_id'], out[PLACE_ODDS_COL]))
        self.assertEqual(got, {'202605040401': 6.0, '202605040402': 9.0})
        self.assertEqual(out['odds_place_max_late'].tolist(), [12.0, 18.0])


class PlaceModelRoundTripTest(unittest.TestCase):
    def test_train_save_load_predict(self) -> None:
        rng = np.random.default_rng(0)
        n_races, field = 400, 10
        n = n_races * field
        df = pd.DataFrame(rng.normal(size=(n, len(FEATURE_COLS))), columns=list(FEATURE_COLS))
        df['race_id'] = np.repeat(np.arange(n_races), field)
        df['date'] = pd.Timestamp('2024-01-01') + pd.to_timedelta(df['race_id'] // 4, unit='D')
        score = df['log_odds_late'] * -1.5 + rng.normal(scale=0.5, size=n)
        df['target_place'] = (score > np.quantile(score, 0.75)).astype('int64')
        with tempfile.TemporaryDirectory() as raw:
            result = train_place_model(df, Path(raw), seeds=(0, 1))
            meta = load_meta(Path(raw))
            model = PlaceModel.load(Path(raw))
            proba = model.predict(df)
        self.assertEqual(meta.seeds, [0, 1])
        self.assertGreater(result.ensemble_auc, 0.8)
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())

    def test_missing_model_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            with self.assertRaises(FileNotFoundError):
                load_meta(Path(raw))


class PlaceStrategyTest(unittest.TestCase):
    def _frame(self, race_id: str, horse: int) -> pd.DataFrame:
        return pd.DataFrame([{
            'race_id': race_id, 'horse_number': horse, 'odds_used': 8.0,
            'pred_proba': 0.2, 'ev': 1.6, 'skip_reason': '',
        }])

    def test_place_intents_and_combine(self) -> None:
        day = dt.date(2026, 9, 22)
        win = SimulatorStrategy().from_frame(self._frame('202605040711', 3), target_date=day)
        place = SimulatorStrategy(bet_type=BetType.PLACE).from_frame(
            self._frame('202605040711', 5), target_date=day
        )
        self.assertEqual(place.intents[0].bet_type, BetType.PLACE)
        self.assertEqual(place.intents[0].strategy_id, PLACE_STRATEGY_ID)
        combined = combine_results([win, place], target_date=day)
        self.assertEqual(
            [(i.bet_type, i.strategy_id) for i in combined.intents],
            [(BetType.PLACE, PLACE_STRATEGY_ID), (BetType.WIN, STRATEGY_ID)],
        )
        self.assertNotEqual(combined.decision_run_id, win.decision_run_id)


class MixedPurchaseTest(unittest.TestCase):
    DAY = dt.date(2026, 9, 22)

    def _strategy(self):
        return combine_results([
            SimulatorStrategy().from_frame(pd.DataFrame([
                {'race_id': '202605040711', 'horse_number': 3, 'skip_reason': ''},
            ]), target_date=self.DAY),
            SimulatorStrategy(bet_type=BetType.PLACE).from_frame(pd.DataFrame([
                {'race_id': '202605040711', 'horse_number': 5, 'skip_reason': ''},
                {'race_id': '202605040711', 'horse_number': 6, 'skip_reason': ''},
            ]), target_date=self.DAY),
        ], target_date=self.DAY)

    def _service(self, root: Path, mode: BuyingMode, browser_factory, **overrides):
        settings = replace(
            make_settings(root, mode),
            allowed_bet_types=frozenset({'win', 'place'}),
            credentials=Credentials('a', 'b', 'c', 'd'),
            **overrides,
        )
        clock = lambda: dt.datetime(2026, 9, 22, 15, 38, 5, tzinfo=race().post_datetime.tzinfo)
        return PurchaseService(
            settings, Ledger(settings.ledger_path), AuditLogger(settings.log_dir),
            browser_factory, clock=clock,
        )

    def _execute(self, service: PurchaseService):
        return service.execute(
            self._strategy(), target_date=self.DAY,
            races={race().race_id: race()}, evidence={race().race_id: evidence()},
        )

    def test_win_and_place_in_same_race_are_prepared_separately(self) -> None:
        browser = PreparingBrowser()
        with tempfile.TemporaryDirectory() as raw:
            report = self._execute(self._service(Path(raw), BuyingMode.DRY_RUN, lambda: browser))
        self.assertEqual(report.dry_run, 3, report)
        # 1回のログインで、券種ごとに確認画面まで進める
        self.assertTrue(browser.logged_in)
        self.assertEqual(sorted(browser.prepared), [('place', 'place'), ('win',)])

    def test_race_limit_counts_both_bet_types(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            report = self._execute(
                self._service(Path(raw), BuyingMode.PAPER, FakeBrowser, max_race_yen=200)
            )
        # 複勝(200円)が先に通り、単勝(100円)は1レース上限200円を超えるので見送り
        self.assertEqual((report.dry_run, report.rejected), (2, 1))


class PreparingBrowser(FakeBrowser):
    def __init__(self) -> None:
        super().__init__()
        self.prepared: list[tuple[str, ...]] = []

    def prepare_bets(self, race_info, intents):
        self.prepared.append(tuple(intent.bet_type.value for intent in intents))
        return super().prepare_bets(race_info, intents)


class BetTypeLabelTest(unittest.TestCase):
    def test_labels(self) -> None:
        def make(kind: BetType) -> BetIntent:
            return BetIntent('202605040711', kind, (1,), 100, 's', 'r')

        self.assertEqual(_bet_type_label((make(BetType.WIN),)), '単勝')
        self.assertEqual(_bet_type_label((make(BetType.PLACE),)), '複勝')
        with self.assertRaises(BrowserContractError):
            _bet_type_label((make(BetType.WIN), make(BetType.PLACE)))


class MainArgsTest(unittest.TestCase):
    def test_default_runs_both_models(self) -> None:
        args = main_module._parse_args([])
        self.assertEqual((args.win, args.place), (True, True))

    def test_select_one_model(self) -> None:
        self.assertEqual(
            (lambda a: (a.win, a.place))(main_module._parse_args(['--win'])), (True, False)
        )
        self.assertEqual(
            (lambda a: (a.win, a.place))(main_module._parse_args(['--place'])), (False, True)
        )
        self.assertEqual(
            (lambda a: (a.win, a.place))(main_module._parse_args(['--win', '--place'])),
            (True, True),
        )


if __name__ == '__main__':
    unittest.main()
