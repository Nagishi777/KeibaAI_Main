"""どんなレースで資金を回収できるのかを調べる。

軸:
  - 投票数（プール規模）
  - レースクラス（G1 ... 新馬）
  - 馬場・距離・頭数・競馬場・馬場状態
5シードのアンサンブル予測を使い、採用条件（EV>=1.2 / 10-150倍 / drift<=-0.05）
を適用したうえでレース属性別に集計する。
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SP = Path(os.environ.get("ANABA_WORK_DIR", ROOT / "output" / "anaba"))
pd.set_option("display.width", 250)
RNG = np.random.default_rng(4242)
SEEDS = [0, 1, 2, 3, 4]

CLASS_ORDER = [
    "新馬",
    "未勝利",
    "1勝クラス",
    "2勝クラス",
    "3勝クラス",
    "オープン",
    "リステッド",
    "G3",
    "G2",
    "G1",
]


def ensemble() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        q = pd.read_feather(SP / f"wf_preds_seed{s}.feather")
        q = q[["race_id", "horse_number", "p"]].rename(columns={"p": f"p{s}"})
        frames.append(q.set_index(["race_id", "horse_number"]))
    P = pd.concat(frames, axis=1)
    P["p"] = P[[f"p{s}" for s in SEEDS]].mean(axis=1)
    base = pd.read_feather(SP / "wf_preds_seed0.feather").drop(columns=["p"])
    out = base.set_index(["race_id", "horse_number"]).join(P[["p"]]).reset_index()
    raw = pd.read_feather(SP / "base.feather")
    attr = raw[
        [
            "race_id",
            "race_class",
            "track_type",
            "distance",
            "racecourse",
            "track_condition",
            "weather",
        ]
    ].drop_duplicates("race_id")
    out = out.merge(attr, on="race_id", how="left")
    out["ev"] = out["p"] * out["odds_1m"]
    return out


def roi_ci(x: pd.DataFrame, n: int = 3000) -> tuple[float, float, float]:
    """レース単位ブートストラップの回収率と95%CI。"""
    st = np.full(len(x), 100.0)
    rt = x["payout_win"].to_numpy(float)
    codes, uniq = pd.factorize(x["race_id"].to_numpy())
    gs = np.bincount(codes, weights=st, minlength=len(uniq))
    gr = np.bincount(codes, weights=rt, minlength=len(uniq))
    roi = gr.sum() / gs.sum()
    if len(uniq) < 5:
        return roi, np.nan, np.nan
    dr = RNG.integers(0, len(uniq), size=(n, len(uniq)))
    rois = gr[dr].sum(1) / gs[dr].sum(1)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return roi, lo, hi


def table(
    b: pd.DataFrame, col: str, min_n: int = 80, order: list | None = None
) -> pd.DataFrame:
    rows = []
    for k, x in b.groupby(col, observed=True):
        if len(x) < min_n:
            continue
        roi, lo, hi = roi_ci(x)
        rows.append(
            {
                col: k,
                "n": len(x),
                "races": x["race_id"].nunique(),
                "hits": int(x["won"].sum()),
                "hit_rate": x["won"].mean(),
                "mean_odds": x["odds_1m"].mean(),
                "roi": roi,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    t = pd.DataFrame(rows)
    if order is not None and len(t):
        t["__o"] = t[col].map({k: i for i, k in enumerate(order)})
        t = t.sort_values("__o").drop(columns="__o")
    elif len(t):
        t = t.sort_values("roi", ascending=False)
    return t


def main() -> None:
    e = ensemble()
    RULE = (e["ev"] >= 1.2) & e["odds_1m"].between(10, 150) & (e["drift_5_1"] <= -0.05)
    b = e[RULE].copy()
    roi = (b["payout_win"] / 100).sum() / len(b)
    print(f"adopted rule: n={len(b)} races={b['race_id'].nunique()} roi={roi:.4f}")

    print("\n" + "=" * 90)
    print("A. 投票数（1m時点のレース総投票額）")
    print("=" * 90)
    # 固定の円しきい値（学習期間の分布から決まる実装可能な値）
    edges = [0, 3e8, 5e8, 7e8, 1e9, 2e9, 1e13]
    labels = ["<3億", "3-5億", "5-7億", "7-10億", "10-20億", "20億+"]
    b["pool_band"] = pd.cut(b["pool_1m_yen"], edges, labels=labels)
    print(table(b, "pool_band", min_n=50, order=labels).round(4).to_string(index=False))

    print("\n--- 参考: 全出走馬ベースの投票数分布 ---")
    e["pool_band"] = pd.cut(e["pool_1m_yen"], edges, labels=labels)
    g = e.groupby("pool_band", observed=True).agg(
        races=("race_id", "nunique"),
        flat_roi=("payout_win", lambda s: s.mean() / 100.0),
    )
    print(g.round(4).to_string())

    print("\n" + "=" * 90)
    print("B. レースクラス（G1 ... 新馬）")
    print("=" * 90)
    print(
        table(b, "race_class", min_n=40, order=CLASS_ORDER)
        .round(4)
        .to_string(index=False)
    )
    print("\n--- クラスを格付け/条件戦でまとめる ---")
    graded = {
        "G1": "graded",
        "G2": "graded",
        "G3": "graded",
        "リステッド": "graded",
        "オープン": "graded",
    }
    b["class_grp"] = b["race_class"].map(graded).fillna("conditional")
    print(table(b, "class_grp", min_n=40).round(4).to_string(index=False))
    print("\n--- 新馬・未勝利だけ ---")
    b["maiden"] = np.where(b["race_class"].isin(["新馬", "未勝利"]), "maiden", "other")
    print(table(b, "maiden", min_n=40).round(4).to_string(index=False))

    print("\n" + "=" * 90)
    print("C. その他の軸")
    print("=" * 90)
    print("\n[馬場]")
    print(table(b, "track_type", min_n=40).round(4).to_string(index=False))
    print("\n[頭数]")
    b["fs_band"] = pd.cut(b["field_size"], [0, 9, 12, 14, 16, 18])
    print(table(b, "fs_band", min_n=40).round(4).to_string(index=False))
    print("\n[距離]")
    b["dist_band"] = pd.cut(
        b["distance"], [0, 1200, 1400, 1600, 1800, 2000, 2400, 5000]
    )
    print(table(b, "dist_band", min_n=40).round(4).to_string(index=False))
    print("\n[競馬場]")
    print(table(b, "racecourse", min_n=80).round(4).to_string(index=False))
    print("\n[馬場状態]")
    print(table(b, "track_condition", min_n=40).round(4).to_string(index=False))

    print("\n" + "=" * 90)
    print("D. 投票数 x クラス（交差）")
    print("=" * 90)
    b["pool_hi"] = np.where(b["pool_1m_yen"] >= 7e8, "pool>=7億", "pool<7億")
    piv = (
        b.pivot_table(
            index="class_grp",
            columns="pool_hi",
            values="payout_win",
            aggfunc="mean",
            observed=True,
        )
        / 100.0
    )
    cnt = b.pivot_table(
        index="class_grp",
        columns="pool_hi",
        values="payout_win",
        aggfunc="size",
        observed=True,
    )
    print("ROI:")
    print(piv.round(3).to_string())
    print("N:")
    print(cnt.to_string())

    print("\n" + "=" * 90)
    print("E. 年ごとに見た主要な軸の安定性")
    print("=" * 90)
    b["year"] = b["date"].dt.year
    for col, lab in (
        ("track_type", "馬場"),
        ("class_grp", "クラス群"),
        ("pool_hi", "投票数"),
    ):
        piv = (
            b.pivot_table(
                index="year",
                columns=col,
                values="payout_win",
                aggfunc="mean",
                observed=True,
            )
            / 100.0
        )
        cnt = b.pivot_table(
            index="year",
            columns=col,
            values="payout_win",
            aggfunc="size",
            observed=True,
        )
        print(f"\n[{lab}] ROI")
        print(piv.round(3).to_string())
        print("N")
        print(cnt.to_string())

    # カテゴリ列は feather に書けないので落としてから保存する
    drop = [c for c in b.columns if str(b[c].dtype) == "category"]
    b.drop(columns=drop).reset_index(drop=True).to_feather(SP / "adopted_bets.feather")
    drop_e = [c for c in e.columns if str(e[c].dtype) == "category"]
    e.drop(columns=drop_e).reset_index(drop=True).to_feather(
        SP / "ensemble_all.feather"
    )
    print("\nsaved adopted_bets.feather / ensemble_all.feather")


if __name__ == "__main__":
    main()
