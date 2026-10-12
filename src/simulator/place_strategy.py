"""穴馬戦略 複勝版の購入判定ルール。

検証内容は ``docs/report/anaba_strategy_place_report/anaba_place_band_report.md``。
採用ルール（同レポート §4・§7.1 の ``bet_rule``）::

    [1] 複勝EV = 予測複勝率 × 判断時点の複勝オッズ(min側) >= 1.05
    [2] 判断時点の複勝オッズ(min側)が 6倍以上 20倍以下
    [3] log(単勝の判断時点オッズ / 単勝5mオッズ) <= -0.05   ← 単勝版と同じドリフト条件
    [4] 条件を満たす馬が複数いるレースでは、期待値の高い順に最大3点

    賭け金は Flat ベット（1点 100円）

IMPORTANT (期待値は必ず複勝オッズの min 側で計算する):
    複勝の配当は同時に入着した馬で変わるため、締切前には min〜max の幅しか
    確定していない。min 側は的中時の実測配当より約18%低く見積もる保守側であり、
    存在しない有利さを織り込まないための選択である（レポート §1.4）。

IMPORTANT (20倍超を買わない):
    20〜30倍で回収率45〜71%、30倍超は2ケース合わせて27点すべて不的中だった
    （レポート §3.2）。上限は「回収率が最大の帯」ではなく、CI下側を最大化する
    事前固定の基準で選ばれた値である。

NOTE (総投票額フィルタは掛けない):
    単勝版の総投票額フィルタは複勝レポートの検証構成に含まれていないため、
    複勝ルールには入れていない。
"""
import logging
from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.simulator.features import DRIFT_COL, LATE
from src.simulator.strategy import (
    REASON_BET,
    REASON_DRIFT,
    REASON_EV,
    REASON_MAX_BETS,
    REASON_ODDS_HIGH,
    REASON_ODDS_LOW,
    REASON_SEP,
    _join_reasons,
)

logger = logging.getLogger(__name__)

# 判断時点の複勝オッズ（min 側 = EV 計算に使う / max 側 = 記録用）
PLACE_ODDS_COL: str = f'odds_place_min_{LATE}'
PLACE_ODDS_MAX_COL: str = f'odds_place_max_{LATE}'

# レポート §7.1 の bet_rule
DEFAULT_MIN_EV: float = 1.05
DEFAULT_ODDS_MIN: float = 6.0
DEFAULT_ODDS_MAX: float = 20.0
DEFAULT_MAX_DRIFT: float = -0.05
DEFAULT_MAX_BETS_PER_RACE: int = 3
DEFAULT_STAKE_YEN: float = 100.0

# 複勝オッズが取れていない馬（補完せず見送る）
REASON_PLACE_ODDS_MISSING = 'place_odds_missing'


@dataclass(frozen=True)
class PlaceRule:
    """複勝の選別ルール。

    Attributes:
        min_ev: 複勝EV（予測複勝率 × 複勝オッズ min 側）の下限
        odds_min: 複勝オッズ（min 側）の下限
        odds_max: 複勝オッズ（min 側）の上限
        max_drift: 単勝の ``log(判断時点オッズ / 5mオッズ)`` の上限
        max_bets_per_race: 1レースあたりの最大点数（期待値上位から採用）
        stake_yen: Flat ベットの1点あたり金額（円）
    """

    min_ev: float = DEFAULT_MIN_EV
    odds_min: float = DEFAULT_ODDS_MIN
    odds_max: float = DEFAULT_ODDS_MAX
    max_drift: float = DEFAULT_MAX_DRIFT
    max_bets_per_race: int = DEFAULT_MAX_BETS_PER_RACE
    stake_yen: float = DEFAULT_STAKE_YEN

    def describe(self) -> str:
        """ログ・CSV のヘッダに出す1行の要約を返す。"""
        return (
            f'複勝EV>={self.min_ev:.2f} / '
            f'複勝オッズ(min) {self.odds_min:g}〜{self.odds_max:g}倍 / '
            f'単勝ドリフト<={self.max_drift:+.2f} / '
            f'1レース{self.max_bets_per_race}点 / '
            f'Flat {self.stake_yen:,.0f}円'
        )


def apply_place_rule(
    df: pd.DataFrame, proba: np.ndarray, rule: Optional[PlaceRule] = None
) -> pd.DataFrame:
    """全馬に ``ev`` / ``skip_reason`` を付けて返す。

    購入しない馬も落とさず理由を付けて返す（「なぜ買わなかったか」を残す）。

    Args:
        df: ``race_id`` / ``PLACE_ODDS_COL`` / ``DRIFT_COL`` を持つ表
        proba: 校正済み予測複勝率（``df`` と同じ行順）
        rule: 選別ルール。省略時は採用値

    Returns:
        pd.DataFrame: ``pred_proba`` / ``odds_used`` / ``ev`` /
            ``skip_reason`` を付けた表（レース・期待値順）

    Raises:
        ValueError: 必須列が欠けている、または予測確率・ドリフトに欠損がある場合
    """
    rule = rule or PlaceRule()
    need = ['race_id', PLACE_ODDS_COL, DRIFT_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f'複勝の選別に必要な列がありません: {missing}。補完せず処理を停止します。'
        )

    work = df.copy().reset_index(drop=True)
    work['pred_proba'] = np.asarray(proba, dtype='float64')
    # 賭け判断オッズは複勝の min 側（保守側）。単勝の判断時点と同じ時点を使う。
    work['odds_used'] = work[PLACE_ODDS_COL].astype('float64')
    work['ev'] = work['pred_proba'] * work['odds_used']

    n_bad = int(work[['pred_proba', DRIFT_COL]].isna().any(axis=1).sum())
    if n_bad:
        raise ValueError(
            f'予測確率・ドリフトに欠損がある行が {n_bad} 件あります。'
            ' 補完せず処理を停止します。'
        )

    no_odds = work['odds_used'].isna()
    work['skip_reason'] = _join_reasons(
        {
            REASON_PLACE_ODDS_MISSING: no_odds,
            REASON_ODDS_LOW: work['odds_used'] < rule.odds_min,
            REASON_ODDS_HIGH: work['odds_used'] > rule.odds_max,
            REASON_DRIFT: work[DRIFT_COL] > rule.max_drift,
            REASON_EV: work['ev'] < rule.min_ev,
        },
        work.index,
    )

    candidate = work['skip_reason'] == REASON_BET
    if candidate.any():
        ranks = (
            work.loc[candidate]
            .groupby('race_id')['ev']
            .rank(ascending=False, method='first')
        )
        dropped = ranks[ranks > rule.max_bets_per_race].index
        work.loc[dropped, 'skip_reason'] = REASON_MAX_BETS

    _log_funnel(work, rule)
    return work.sort_values(
        ['race_id', 'ev'], ascending=[True, False], na_position='last'
    ).reset_index(drop=True)


def _log_funnel(work: pd.DataFrame, rule: PlaceRule) -> None:
    """どのフィルタで何頭落ちたかをログに出す（当日の確認用）。"""
    reasons = work['skip_reason'].str.split(REASON_SEP)
    counts = {
        label: int(reasons.apply(lambda r, label=label: label in r).sum())
        for label in (
            REASON_PLACE_ODDS_MISSING, REASON_ODDS_LOW, REASON_ODDS_HIGH,
            REASON_DRIFT, REASON_EV, REASON_MAX_BETS,
        )
    }
    n_bets = int((work['skip_reason'] == REASON_BET).sum())
    logger.warning('[複勝] 購入条件: %s', rule.describe())
    logger.warning(
        '[複勝] 選別結果: %s 頭 / %s レース → 購入 %s 点'
        '（内訳: 複勝オッズ欠損 %s / オッズ下限 %s / オッズ上限 %s /'
        ' ドリフト %s / EV %s / レース内上位外 %s）',
        f'{len(work):,}', f'{work["race_id"].nunique():,}', f'{n_bets:,}',
        counts[REASON_PLACE_ODDS_MISSING], counts[REASON_ODDS_LOW],
        counts[REASON_ODDS_HIGH], counts[REASON_DRIFT], counts[REASON_EV],
        counts[REASON_MAX_BETS],
    )


def place_rule_from_config(
    config: dict, *, min_ev: Optional[float] = None
) -> Tuple[PlaceRule, Dict[str, object]]:
    """``config.yaml`` の ``place_strategy`` セクションからルールを作る。

    Args:
        config: 全体設定辞書
        min_ev: 指定すれば EV しきい値を上書きする（コマンドライン用）

    Returns:
        Tuple[PlaceRule, Dict[str, object]]: (ルール, 使った設定値)
    """
    cfg = config.get('place_strategy', {}) or {}
    rule = PlaceRule(
        min_ev=float(cfg.get('min_ev', DEFAULT_MIN_EV)),
        odds_min=float(cfg.get('odds_min', DEFAULT_ODDS_MIN)),
        odds_max=float(cfg.get('odds_max', DEFAULT_ODDS_MAX)),
        max_drift=float(cfg.get('max_drift', DEFAULT_MAX_DRIFT)),
        max_bets_per_race=int(
            cfg.get('max_bets_per_race', DEFAULT_MAX_BETS_PER_RACE)
        ),
        stake_yen=float(cfg.get('stake_yen', DEFAULT_STAKE_YEN)),
    )
    if min_ev is not None:
        rule = replace(rule, min_ev=float(min_ev))
    return rule, dict(cfg)
