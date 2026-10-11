"""Feature construction for the steam-move / longshot overlay model.

IMPORTANT: every column here uses only pre-close market snapshots
(60m/30m/10m/5m/1m). No finishing position, final odds, popularity or payout
is used as an input. Race attributes (distance/track/class/field size) are
known before the off.
"""

import numpy as np
import pandas as pd

TAKEOUT = 0.20
EPS = 1e-12
SNAPS = ("60m", "30m", "10m", "5m", "1m")


def estimate_votes(odds: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """Recover per-horse stake from win odds and race pool (JRA odds formula)."""
    valid = (odds > EPS) & (pool > EPS) & np.isfinite(odds) & np.isfinite(pool)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(
            valid,
            (1.0 - TAKEOUT) * np.where(valid, pool, 0.0) / np.where(valid, odds, 1.0),
            np.nan,
        )
    return v.astype("float64")


def _race_sum(df: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    tmp = pd.DataFrame({"r": df["race_id"].to_numpy(), "v": values})
    return tmp.groupby("r", sort=False)["v"].transform("sum").to_numpy(dtype="float64")


def build(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # --- per-snapshot stakes and market probability shares ---
    for s in SNAPS:
        d["votes_" + s] = estimate_votes(
            d["odds_" + s].to_numpy(dtype="float64"),
            d["pool_" + s].to_numpy(dtype="float64"),
        )
        inv = 1.0 / d["odds_" + s].to_numpy(dtype="float64")
        tot = _race_sum(d, np.nan_to_num(inv, nan=0.0))
        d["mkt_" + s] = inv / np.where(tot > EPS, tot, np.nan)

    # --- log-odds drift over each interval (the core signal) ---
    d["drift_5_1"] = np.log(d["odds_1m"] / d["odds_5m"])
    d["drift_10_5"] = np.log(d["odds_5m"] / d["odds_10m"])
    d["drift_30_10"] = np.log(d["odds_10m"] / d["odds_30m"])
    d["drift_60_30"] = np.log(d["odds_30m"] / d["odds_60m"])
    d["drift_60_1"] = np.log(d["odds_1m"] / d["odds_60m"])
    # acceleration: is the move fresh or a continuation?
    d["accel"] = d["drift_5_1"] - d["drift_10_5"]

    # --- drift relative to the race (a race can drift as a whole) ---
    for c in ("drift_5_1", "drift_10_5"):
        m = d.groupby("race_id")[c].transform("median")
        d[c + "_rel"] = d[c] - m

    # --- incremental money share in the final interval ---
    inc = (d["votes_1m"] - d["votes_5m"]).to_numpy(dtype="float64")
    inc = np.where(np.isfinite(inc), np.maximum(inc, 0.0), np.nan)
    tot_inc = _race_sum(d, np.nan_to_num(inc, nan=0.0))
    d["inc_share_5_1"] = inc / np.where(tot_inc > EPS, tot_inc, np.nan)
    # is this horse taking a bigger share of new money than of standing money?
    d["inc_vs_mkt"] = d["inc_share_5_1"] - d["mkt_5m"]
    d["inc_ratio"] = d["inc_share_5_1"] / d["mkt_5m"].where(d["mkt_5m"] > EPS)

    # --- levels ---
    d["log_odds_1m"] = np.log(d["odds_1m"])
    d["log_odds_5m"] = np.log(d["odds_5m"])
    d["mkt_rank_1m"] = d.groupby("race_id")["odds_1m"].rank(method="min")
    d["pool_1m_yen"] = d["pool_1m"] * 100.0
    d["log_pool_1m"] = np.log(d["pool_1m_yen"])
    d["pool_growth_5_1"] = d["pool_1m"] / d["pool_5m"]
    d["pool_per_horse"] = d["pool_1m_yen"] / d["field_size"]
    d["log_pool_per_horse"] = np.log(d["pool_per_horse"])

    # --- race-level dispersion ---
    d["hhi_1m"] = d.groupby("race_id")["mkt_1m"].transform(lambda s: (s**2).sum())
    fav = d.groupby("race_id")["mkt_1m"].transform("max")
    d["fav_strength"] = fav
    d["mkt_ratio_to_fav"] = d["mkt_1m"] / fav

    # --- race attributes (known pre-race) ---
    d["is_turf"] = (d["track_type"].astype(str).str.contains("芝")).astype("int8")
    d["is_dirt"] = (d["track_type"].astype(str).str.contains("ダ")).astype("int8")
    age = d["sex_age"].astype(str).str.extract(r"(\d+)")[0]
    d["age"] = pd.to_numeric(age, errors="coerce")
    d["impost"] = pd.to_numeric(d["impost"], errors="coerce")
    d["horse_weight"] = pd.to_numeric(d["horse_weight"], errors="coerce")
    d["horse_weight_change"] = pd.to_numeric(d["horse_weight_change"], errors="coerce")

    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    return d


# 29列の全特徴量。歴史的な既定値で、検証もこの構成で行った。
#
# NOTE (特徴量の削減について / docs/anaba_feature_ablation_report.md):
#     gain importance の 89.9% を上位4列が占め、下位15列は合計2%未満である。
#     8列に削っても 8シードアンサンブルの回収率は 139.9% で、
#     29列の 139.8% と差がない（対応ブートストラップの95%CIが0を跨ぐ）。
#     検証AUCも 0.8432 対 0.8431 でほぼ同一。
#     実運用で列を減らしたい場合は ``FEATURES_TOP8`` を使う。
FEATURES = [
    "log_odds_1m",
    "log_odds_5m",
    "mkt_1m",
    "mkt_rank_1m",
    "drift_5_1",
    "drift_10_5",
    "drift_30_10",
    "drift_60_30",
    "drift_60_1",
    "accel",
    "drift_5_1_rel",
    "drift_10_5_rel",
    "inc_share_5_1",
    "inc_vs_mkt",
    "inc_ratio",
    "log_pool_1m",
    "pool_growth_5_1",
    "log_pool_per_horse",
    "hhi_1m",
    "fav_strength",
    "mkt_ratio_to_fav",
    "field_size",
    "distance",
    "is_turf",
    "is_dirt",
    "age",
    "impost",
    "horse_weight",
    "horse_weight_change",
]


# 削減版（8列）。``FEATURES`` と同等の回収率・AUC を保ちつつ、
# 計算量と保守対象を減らしたい場合に使う。
# 内訳は gain importance 上位4列に、相関の低い補助列を4本足したもの。
FEATURES_TOP8 = [
    # --- 中核（gain の 89.9% を占める4列）---
    "inc_share_5_1",  # 直前4分の新規資金シェア（単独で gain の65%）
    "log_odds_1m",  # 1分前オッズの水準
    "mkt_1m",  # 1分前の市場確率シェア
    "mkt_ratio_to_fav",  # 1番人気に対する相対的な強さ
    # --- 補助（レース内の文脈とノイズ補正）---
    "horse_weight",  # 馬体重
    "drift_30_10",  # 30分前→10分前のオッズ変化
    "accel",  # ドリフトの加速（直前の動きが新しいか）
    "drift_10_5_rel",  # 10分前→5分前のドリフト（レース中央値からの差）
]
