"""穴馬（高オッズ・高期待値）戦略の選別ルールと賭け金計算。

戦略の中身は「締切直前4分間に資金が集まった中オッズ〜高オッズ馬を買う」。
詳細と検証結果は ``docs/anaba_strategy_report.md`` を参照。

IMPORTANT (時点の一貫性):
    選別に使う情報は 1m スナップショットまで、賭け判断オッズも 1m を使う。
    確定オッズ・着順・払戻は選別に一切使わない。
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# 検証で採用したしきい値（docs/anaba_strategy_report.md §5）
DEFAULT_MIN_EV: float = 1.20
DEFAULT_ODDS_MIN: float = 10.0
DEFAULT_ODDS_MAX: float = 150.0
DEFAULT_MAX_DRIFT: float = -0.05
DEFAULT_MAX_BETS_PER_RACE: int = 3


@dataclass(frozen=True)
class AnabaRule:
    """穴馬選別ルール。

    Attributes:
        min_ev: 期待値の下限（``予測確率 × 1mオッズ``）
        odds_min: 1mオッズの下限。これ未満は妙味が小さい
        odds_max: 1mオッズの上限。これ超は的中率が落ちて期待値が崩れる
        max_drift: ``log(odds_1m / odds_5m)`` の上限。
            負値なのでオッズが下がった（資金が入った）馬だけが残る
        max_bets_per_race: 1レースあたりの最大点数（期待値上位から採用）
    """

    min_ev: float = DEFAULT_MIN_EV
    odds_min: float = DEFAULT_ODDS_MIN
    odds_max: float = DEFAULT_ODDS_MAX
    max_drift: float = DEFAULT_MAX_DRIFT
    max_bets_per_race: int = DEFAULT_MAX_BETS_PER_RACE


def select(df: pd.DataFrame, rule: AnabaRule | None = None) -> pd.DataFrame:
    """予測済みデータフレームから購入対象を選別する。

    Args:
        df: ``p``（校正済み勝率）・``odds_1m``・``odds_5m``・``race_id`` を持つ表。
            ``drift_5_1`` があればそれを使い、無ければ2つのオッズから計算する
        rule: 選別ルール。省略時は検証で採用した既定値

    Returns:
        pd.DataFrame: 購入対象の行のみ。``ev`` 列と ``ev_rank`` 列を付与する

    Raises:
        ValueError: 必須列が欠けている場合（補完せず停止する）
    """
    rule = rule or AnabaRule()
    need = ["race_id", "p", "odds_1m", "odds_5m"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f"穴馬選別に必要な列がありません: {missing}。補完せず処理を停止します。"
        )

    out = df.copy()
    if "drift_5_1" not in out.columns:
        out["drift_5_1"] = np.log(out["odds_1m"] / out["odds_5m"])

    n_bad = int(out[["p", "odds_1m", "drift_5_1"]].isna().any(axis=1).sum())
    if n_bad:
        raise ValueError(
            f"予測確率・オッズ・ドリフトに欠損がある行が {n_bad} 件あります。"
            " 補完せず処理を停止します。"
        )

    out["ev"] = out["p"] * out["odds_1m"]
    mask = (
        (out["ev"] >= rule.min_ev)
        & (out["odds_1m"] >= rule.odds_min)
        & (out["odds_1m"] <= rule.odds_max)
        & (out["drift_5_1"] <= rule.max_drift)
    )
    picked = out.loc[mask].copy()
    picked["ev_rank"] = picked.groupby("race_id")["ev"].rank(
        ascending=False, method="first"
    )
    picked = picked.loc[picked["ev_rank"] <= rule.max_bets_per_race]
    # 馬券購入時に上から順に扱えるよう、レース内は期待値の降順で返す
    picked = picked.sort_values(["race_id", "ev"], ascending=[True, False])
    return picked.reset_index(drop=True)


def kelly_stake(
    p: np.ndarray,
    odds: np.ndarray,
    bankroll: float,
    *,
    fraction: float = 0.10,
    unit: float = 100.0,
    max_fraction_per_bet: float = 0.02,
) -> np.ndarray:
    """fractional Kelly で賭け金を決める（``unit`` 円単位に切り上げる）。

    Kelly の最適比率は ``f* = (p * o - 1) / (o - 1)``。
    穴馬戦略は的中率5%・平均40倍の高分散なので、``fraction`` を小さく取り
    さらに1点あたりの上限を設けて破産確率を抑える。

    IMPORTANT (切り捨てで点数を失わないこと):
        単純に ``floor`` で丸めると、資金が小さいときに Kelly 額が
        ``unit`` 未満になった点が全て 0 円になり、**戦略の一部の馬が
        黙って買われなくなる**。それでは検証した回収率と別物になるので、
        期待値が正の点は最低 ``unit`` 円を賭ける（切り上げる）。
        資金がそれに耐えられない場合は例外を投げて停止する。

    Args:
        p: 校正済み勝率
        odds: 賭け判断オッズ（1m時点）
        bankroll: 現在の資金
        fraction: Kelly比率に掛ける係数（既定 0.10）
        unit: 購入単位（円）
        max_fraction_per_bet: 1点あたりの資金上限比率

    Returns:
        np.ndarray: 各点の賭け金（円）。期待値が負の点は 0

    Raises:
        ValueError: 1点あたりの上限が購入単位を下回る資金しかない場合
    """
    p = np.asarray(p, dtype="float64")
    odds = np.asarray(odds, dtype="float64")
    if bankroll * max_fraction_per_bet < unit:
        raise ValueError(
            f"資金 {bankroll:.0f} 円では1点あたり上限"
            f"（{bankroll * max_fraction_per_bet:.0f} 円）が購入単位"
            f" {unit:.0f} 円に届きません。"
            f" 資金を {unit / max_fraction_per_bet:.0f} 円以上にしてください。"
        )
    edge = p * odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f_star = np.where(odds > 1.0, edge / (odds - 1.0), 0.0)
    f = np.clip(f_star * fraction, 0.0, max_fraction_per_bet)
    # 期待値が正の点は最低1単位。切り上げで「買わない」を作らない。
    stake = np.ceil(bankroll * f / unit) * unit
    stake = np.minimum(stake, np.floor(bankroll * max_fraction_per_bet / unit) * unit)
    return np.where(edge > 0.0, np.maximum(stake, unit), 0.0)
