"""レポート用の図を作る。ラベルは全て英語（CLAUDE.md の規約）。"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[3]
SP = Path(os.environ.get("ANABA_WORK_DIR", ROOT / "output" / "anaba"))
IMG = ROOT / "docs" / "img"
IMG.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
INK = "#1c1c1c"
GRID = "#d8d8d8"
POS = "#1f6f4a"
NEG = "#a8324a"
ACC = "#2f5d8c"
BREAK = "#b0442e"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.edgecolor": "#666666",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def load_base():
    return pd.read_feather(SP / "base.feather")


def load_ensemble():
    return pd.read_feather(SP / "ensemble_all.feather")


def load_bets():
    return pd.read_feather(SP / "adopted_bets.feather")


def roi_ci(x, n=2500, seed=11):
    """レース単位ブートストラップの回収率と95%CI。"""
    rng = np.random.default_rng(seed)
    rt = x["payout_win"].to_numpy(float)
    codes, uniq = pd.factorize(x["race_id"].to_numpy())
    gs = np.bincount(codes, weights=np.full(len(x), 100.0), minlength=len(uniq))
    gr = np.bincount(codes, weights=rt, minlength=len(uniq))
    roi = gr.sum() / gs.sum()
    if len(uniq) < 5:
        return roi, roi, roi
    dr = rng.integers(0, len(uniq), size=(n, len(uniq)))
    rois = gr[dr].sum(1) / gs[dr].sum(1)
    lo, hi = np.percentile(rois, [2.5, 97.5])
    return roi, lo, hi


# ----------------------------------------------------------------------------
# FIG 1: the crossing point — drift x odds band heatmap
# ----------------------------------------------------------------------------
def fig_crossing():
    d = load_base()
    d = d[d["odds_5m"].notna() & d["odds_1m"].notna()].copy()
    d["drift"] = np.log(d["odds_1m"] / d["odds_5m"])
    d["ob"] = pd.cut(
        d["odds_1m"],
        [0, 5, 10, 20, 50, 150, 1e9],
        labels=["1-5", "5-10", "10-20", "20-50", "50-150", "150+"],
    )
    d["dq"] = pd.qcut(
        d["drift"],
        5,
        labels=[
            "strong inflow\n(odds crash)",
            "inflow",
            "flat",
            "outflow",
            "strong outflow",
        ],
    )
    piv = (
        d.pivot_table(
            index="dq", columns="ob", values="payout_win", aggfunc="mean", observed=True
        )
        / 100.0
    )
    cnt = d.pivot_table(
        index="dq", columns="ob", values="payout_win", aggfunc="size", observed=True
    )

    cmap = LinearSegmentedColormap.from_list(
        "roi", ["#8c2f42", "#c98a94", "#f2f2f2", "#8fbfa4", "#1f6f4a"]
    )
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    vals = piv.to_numpy(dtype=float)
    im = ax.imshow(vals, cmap=cmap, vmin=0.3, vmax=2.2, aspect="auto")

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = vals[i, j]
            if not np.isfinite(v):
                continue
            n = cnt.to_numpy()[i, j]
            ax.text(
                j,
                i - 0.13,
                f"{v * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold" if v > 1.0 else "normal",
                color="white" if (v < 0.55 or v > 1.75) else INK,
            )
            ax.text(
                j,
                i + 0.22,
                f"n={n:,}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if (v < 0.55 or v > 1.75) else "#666666",
            )

    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, fontsize=9)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_xlabel("Win odds at 1 minute before post", fontsize=10)
    ax.set_ylabel("Direction of money in the last 4 minutes", fontsize=10)
    ax.set_title(
        "The crossing point: ROI by money flow x odds band\n"
        "Flat 100 yen on every runner, 2023-01 to 2026-07 "
        "(165,714 runners). Break-even = 100%",
        fontsize=11,
        loc="left",
        pad=12,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("ROI", fontsize=9)
    cb.ax.axhline(1.0, color=BREAK, lw=1.6)
    # highlight the profitable block
    ax.add_patch(
        plt.Rectangle((1.5, -0.5), 4, 1, fill=False, edgecolor=POS, lw=2.4, zorder=5)
    )
    ax.text(
        2.0,
        -0.62,
        "the edge: long odds + late money",
        color=POS,
        fontsize=9.5,
        fontweight="bold",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(IMG / "anaba_crossing_heatmap.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig1 ok")


# ----------------------------------------------------------------------------
# FIG 2: why naive longshot betting fails + what drift adds
# ----------------------------------------------------------------------------
def fig_oddsband():
    d = load_base()
    d = d[d["odds_1m"].notna() & d["odds_5m"].notna()].copy()
    d["drift"] = np.log(d["odds_1m"] / d["odds_5m"])
    bins = [0, 2, 3, 5, 7, 10, 15, 20, 30, 50, 80, 150, 1e9]
    lab = [
        "1-2",
        "2-3",
        "3-5",
        "5-7",
        "7-10",
        "10-15",
        "15-20",
        "20-30",
        "30-50",
        "50-80",
        "80-150",
        "150+",
    ]
    d["ob"] = pd.cut(d["odds_1m"], bins, labels=lab)
    allr = d.groupby("ob", observed=True)["payout_win"].mean() / 100.0
    fall = (
        d[d["drift"] <= -0.05].groupby("ob", observed=True)["payout_win"].mean() / 100.0
    )
    rise = d[d["drift"] > 0].groupby("ob", observed=True)["payout_win"].mean() / 100.0

    x = np.arange(len(lab))
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.axhline(1.0, color=BREAK, lw=1.6, zorder=1)
    ax.text(
        len(lab) - 0.4,
        1.02,
        "break-even 100%",
        color=BREAK,
        fontsize=8.5,
        ha="right",
        va="bottom",
    )
    ax.axhline(0.8, color="#999999", lw=1.0, ls=":", zorder=1)
    ax.text(0.05, 0.805, "takeout floor 80%", color="#777777", fontsize=8, va="bottom")

    ax.plot(
        x,
        allr.reindex(lab).to_numpy(),
        "o-",
        color="#8a8a8a",
        lw=1.8,
        ms=5,
        label="All runners",
        zorder=3,
    )
    ax.plot(
        x,
        fall.reindex(lab).to_numpy(),
        "o-",
        color=POS,
        lw=2.4,
        ms=6,
        label="Odds fell in last 4 min (money coming in)",
        zorder=4,
    )
    ax.plot(
        x,
        rise.reindex(lab).to_numpy(),
        "o-",
        color=NEG,
        lw=1.8,
        ms=5,
        label="Odds rose (money leaving)",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=9)
    ax.set_xlabel("Win odds at 1 minute before post", fontsize=10)
    ax.set_ylabel("ROI (flat bet)", fontsize=10)
    ax.set_ylim(0.2, 2.4)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(
        "Longshots alone lose. Longshots with late money win.\n"
        "The two lines separate above 10x and never re-cross",
        fontsize=11,
        loc="left",
        pad=10,
    )
    fig.tight_layout()
    fig.savefig(IMG / "anaba_odds_band_roi.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig2 ok")


# ----------------------------------------------------------------------------
# FIG 3: bankroll curves (flat + Kelly) and drawdown
# ----------------------------------------------------------------------------
def fig_bankroll():
    b = load_bets().sort_values(["date", "race_id"]).reset_index(drop=True)

    def kelly_curve(frac):
        bank = 100000.0
        out = []
        peak, maxdd = bank, 0.0
        for _, grp in b.groupby("race_id", sort=False):
            tot = ret = 0.0
            for _, row in grp.iterrows():
                o, pr = row["odds_1m"], row["p"]
                edge = pr * o - 1.0
                if edge <= 0:
                    continue
                kf = min(edge / (o - 1.0) * frac, 0.02)
                stake = max(100.0, np.ceil(bank * kf / 100.0) * 100.0)
                if stake > bank - tot:
                    continue
                tot += stake
                ret += stake * row["payout_win"] / 100.0
            bank += ret - tot
            peak = max(peak, bank)
            maxdd = max(maxdd, (peak - bank) / peak)
            out.append((grp["date"].iloc[0], bank))
        return pd.DataFrame(out, columns=["date", "bank"]), maxdd

    fig, (ax0, ax1, ax2) = plt.subplots(
        3,
        1,
        figsize=(9.6, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 1.55, 0.85], "hspace": 0.30},
    )

    # --- panel 0: flat staking, in yen ---
    pnl = (b["payout_win"] * 10.0 - 1000.0).cumsum()
    ax0.plot(b["date"], pnl / 1e4, color=ACC, lw=1.8, zorder=3)
    ax0.axhline(0, color="#888888", lw=1.0)
    ax0.fill_between(
        b["date"], 0, pnl / 1e4, where=(pnl >= 0), color=ACC, alpha=0.12, lw=0
    )
    ax0.fill_between(
        b["date"], 0, pnl / 1e4, where=(pnl < 0), color=NEG, alpha=0.16, lw=0
    )
    ax0.set_ylabel("Cumulative profit\n(10k yen)", fontsize=9.5)
    ax0.grid(axis="y", color=GRID, lw=0.7)
    ax0.set_axisbelow(True)
    roi_pct = (b["payout_win"] * 10).sum() / (len(b) * 1000) * 100
    ax0.set_title(
        "Flat staking: 1,000 yen on every bet   "
        f"(final +{pnl.iloc[-1]:,.0f} yen, ROI {roi_pct:.0f}%)",
        fontsize=10.5,
        loc="left",
    )
    ax0.annotate(
        "first 9 months are flat to negative",
        xy=(b["date"].iloc[430], pnl.iloc[430] / 1e4),
        xytext=(b["date"].iloc[560], 48),
        fontsize=8.5,
        color=NEG,
        arrowprops={"arrowstyle": "->", "color": NEG, "lw": 1.1},
    )

    # --- panel 1: fractional Kelly, as a multiple of the bankroll ---
    for frac, col, ls in (
        (0.05, "#6b8fb3", ":"),
        (0.10, POS, "-"),
        (0.25, "#c98a3a", "--"),
    ):
        cv, mdd = kelly_curve(frac)
        mult = cv["bank"].iloc[-1] / 1e5
        ax1.plot(
            cv["date"],
            cv["bank"] / 100000.0,
            color=col,
            lw=1.9,
            ls=ls,
            label=f"Kelly {frac:.2f}   x{mult:.1f}   max DD {mdd * 100:.0f}%",
        )
    ax1.axhline(1.0, color="#888888", lw=1.0)
    ax1.set_ylabel("Bankroll\n(multiple of start)", fontsize=9.5)
    ax1.set_yscale("log")
    ax1.set_yticks([1, 2, 5, 10, 20])
    ax1.set_yticklabels(["x1", "x2", "x5", "x10", "x20"])
    ax1.grid(axis="y", color=GRID, lw=0.7)
    ax1.set_axisbelow(True)
    ax1.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax1.set_title(
        "Fractional Kelly: compounding, start 100,000 yen (log scale)",
        fontsize=10.5,
        loc="left",
    )

    # --- panel 2: drawdown of the flat curve ---
    dd = (pnl.cummax() - pnl) / 1e4
    ax2.fill_between(b["date"], 0, -dd, color=NEG, alpha=0.65, lw=0)
    ax2.set_ylabel("Drawdown\n(10k yen)", fontsize=9.5)
    ax2.set_ylim(-dd.max() * 1.25, 0)
    ax2.grid(axis="y", color=GRID, lw=0.7)
    ax2.set_axisbelow(True)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_title(
        f"Drawdown under flat staking: worst {dd.max() * 1e4:,.0f} yen, "
        "longest losing run 116 bets",
        fontsize=10,
        loc="left",
    )

    fig.suptitle(
        "Bankroll path of the adopted strategy: 3,039 bets over "
        "2,318 races, 2024-01 to 2026-07 (out-of-sample)",
        fontsize=11.5,
        x=0.055,
        ha="left",
        y=0.98,
    )
    fig.savefig(IMG / "anaba_bankroll.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"fig3 ok  flat_pnl={pnl.iloc[-1]:.0f}")


# ----------------------------------------------------------------------------
# FIG 4: which races pay — pool size and class
# ----------------------------------------------------------------------------
def fig_racetype():
    b = load_bets()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3))

    # -- pool size --
    edges = [0, 3e8, 5e8, 7e8, 1e9, 2e9, 1e13]
    lab = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-1.0", "1.0-2.0", "2.0+"]
    b = b.copy()
    b["pb"] = pd.cut(b["pool_1m_yen"], edges, labels=lab)
    rows = []
    for k, x in b.groupby("pb", observed=True):
        if len(x) < 50:
            continue
        r, lo, hi = roi_ci(x)
        rows.append({"k": k, "n": len(x), "roi": r, "lo": lo, "hi": hi})
    g = pd.DataFrame(rows).set_index("k")
    ax = axes[0]
    cols = [POS if v > 1 else NEG for v in g["roi"]]
    ax.bar(range(len(g)), g["roi"], color=cols, width=0.66)
    ax.errorbar(
        range(len(g)),
        g["roi"],
        yerr=[g["roi"] - g["lo"], g["hi"] - g["roi"]],
        fmt="none",
        ecolor="#3a3a3a",
        elinewidth=1.2,
        capsize=4,
    )
    ax.axhline(1.0, color=BREAK, lw=1.6)
    for i, (v, n, hi) in enumerate(zip(g["roi"], g["n"], g["hi"])):
        ax.text(
            i,
            hi + 0.06,
            f"{v * 100:.0f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.text(i, 0.06, f"n={n}", ha="center", fontsize=7.5, color="white")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, fontsize=8.5)
    ax.set_xlabel("Race pool at 1 min (billion yen)", fontsize=9.5)
    ax.set_ylabel("ROI", fontsize=9.5)
    ax.set_ylim(0, 3.0)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Thicker betting pool pays more", fontsize=10.5, loc="left")

    # -- class --
    order = [
        "Newcomer",
        "Maiden",
        "Class 1",
        "Class 2",
        "Class 3",
        "Open/List",
        "G3/G2",
        "G1",
    ]
    jp = {
        "新馬": "Newcomer",
        "未勝利": "Maiden",
        "1勝クラス": "Class 1",
        "2勝クラス": "Class 2",
        "3勝クラス": "Class 3",
        "オープン": "Open/List",
        "リステッド": "Open/List",
        "G3": "G3/G2",
        "G2": "G3/G2",
        "G1": "G1",
    }
    b["cl"] = b["race_class"].map(jp)
    rows = []
    for k, x in b.groupby("cl", observed=True):
        r, lo, hi = roi_ci(x)
        rows.append({"k": k, "n": len(x), "roi": r, "lo": lo, "hi": hi})
    g2 = pd.DataFrame(rows).set_index("k")
    g2 = g2.reindex([c for c in order if c in g2.index])
    ax = axes[1]
    cols = [POS if v > 1 else NEG for v in g2["roi"]]
    ax.bar(range(len(g2)), g2["roi"], color=cols, width=0.66)
    ax.errorbar(
        range(len(g2)),
        g2["roi"],
        yerr=[g2["roi"] - g2["lo"], g2["hi"] - g2["roi"]],
        fmt="none",
        ecolor="#3a3a3a",
        elinewidth=1.2,
        capsize=4,
    )
    ax.axhline(1.0, color=BREAK, lw=1.6)
    for i, (v, n, hi) in enumerate(zip(g2["roi"], g2["n"], g2["hi"])):
        ax.text(
            i,
            hi + 0.08,
            f"{v * 100:.0f}%",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.text(i, 0.06, f"n={n}", ha="center", fontsize=7.5, color="white")
    ax.set_xticks(range(len(g2)))
    ax.set_xticklabels(g2.index, fontsize=8.5, rotation=30, ha="right")
    ax.set_xlabel("Race class", fontsize=9.5)
    ax.set_ylim(0, 6.2)
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.set_title("Graded races pay; G1 and newcomers do not", fontsize=10.5, loc="left")

    fig.suptitle(
        "Which races return the money (adopted rule, 3,039 bets)"
        "\nBars are ROI; whiskers are 95% bootstrap intervals",
        fontsize=11,
        x=0.055,
        ha="left",
        y=1.06,
    )
    fig.tight_layout()
    fig.savefig(IMG / "anaba_race_type.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig4 ok")


# ----------------------------------------------------------------------------
# FIG 5: robustness — per-seed / per-year / shuffle / concentration
# ----------------------------------------------------------------------------
def fig_robust():
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7))

    # (a) yearly
    b = load_bets()
    b = b.copy()
    b["year"] = b["date"].dt.year
    g = b.groupby("year").agg(
        n=("won", "size"), roi=("payout_win", lambda s: s.mean() / 100.0)
    )
    ax = axes[0]
    ax.bar(g.index.astype(str), g["roi"], color=POS, width=0.58)
    ax.axhline(1.0, color=BREAK, lw=1.6)
    for i, (v, n) in enumerate(zip(g["roi"], g["n"])):
        ax.text(
            i, v + 0.03, f"{v * 100:.0f}%", ha="center", fontsize=9.5, fontweight="bold"
        )
        ax.text(i, 0.06, f"n={n}", ha="center", fontsize=7.5, color="white")
    ax.set_ylim(0, 1.95)
    ax.set_ylabel("ROI", fontsize=9.5)
    ax.set_title("Every year clears 100%", fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    # (b) shuffle test
    ax = axes[1]
    rng = np.random.default_rng(99)
    e = load_ensemble()
    e["ev"] = e["p"] * e["odds_1m"]
    m = (e["ev"] >= 1.2) & e["odds_1m"].between(10, 150) & (e["drift_5_1"] <= -0.05)
    real = (e.loc[m, "payout_win"] / 100).sum() / m.sum()
    rois = []
    for _ in range(400):
        u = rng.random(len(e))
        wp = e.groupby("race_id")["payout_win"].transform("max")
        pick = e.groupby("race_id")["u" if "u" in e else "payout_win"].transform("max")
        tmp = e.assign(u=u)
        pick = tmp.groupby("race_id")["u"].transform("max") == tmp["u"]
        rois.append((np.where(pick, wp, 0.0)[m.to_numpy()] / 100).sum() / m.sum())
    rois = np.array(rois)
    ax.hist(rois, bins=34, color="#9aa7b4", edgecolor="white", lw=0.5)
    ax.axvline(real, color=POS, lw=2.4)
    ax.text(
        real,
        ax.get_ylim()[1] * 0.92,
        f" actual\n {real * 100:.0f}%",
        color=POS,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.axvline(1.0, color=BREAK, lw=1.3, ls="--")
    ax.set_xlabel("ROI", fontsize=9.5)
    ax.set_ylabel("count", fontsize=9.5)
    ax.set_title("400 label shuffles never reach it", fontsize=10, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    # (c) concentration
    ax = axes[2]
    ret = (load_bets()["payout_win"] / 100.0).sort_values(ascending=False)
    n = len(ret)
    ks = [0, 1, 3, 5, 10, 15, 20]
    vals = [(ret.sum() - ret.head(k).sum()) / n for k in ks]
    ax.plot(range(len(ks)), vals, "o-", color=ACC, lw=2.0, ms=6)
    ax.axhline(1.0, color=BREAK, lw=1.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v * 100:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"-{k}" if k else "all" for k in ks], fontsize=9)
    ax.set_xlabel("Biggest winning bets removed", fontsize=9.5)
    ax.axhspan(0.85, 1.0, color=NEG, alpha=0.07, lw=0)
    ax.set_ylim(0.85, 1.5)
    ax.set_title(
        "Survives losing the top 10 wins,\nbut not the top 20", fontsize=10, loc="left"
    )
    ax.grid(axis="y", color=GRID, lw=0.7)
    ax.set_axisbelow(True)

    fig.suptitle(
        "Robustness checks on the adopted strategy",
        fontsize=11.5,
        x=0.045,
        ha="left",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(IMG / "anaba_robustness.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(
        f"fig5 ok  shuffle mean={rois.mean():.3f} max={rois.max():.3f} p={(rois >= real).mean():.4f}"
    )


# ----------------------------------------------------------------------------
# FIG 6: calibration
# ----------------------------------------------------------------------------
def fig_calib():
    e = load_ensemble()
    e = e.copy()
    e["pb"] = pd.qcut(e["p"], 14, duplicates="drop")
    g = e.groupby("pb", observed=True).agg(
        pred=("p", "mean"), act=("won", "mean"), n=("won", "size")
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    lim = [0.0008, 0.55]
    ax.plot(lim, lim, color="#999999", lw=1.2, ls="--", label="perfect")
    ax.scatter(
        g["pred"],
        g["act"],
        s=np.sqrt(g["n"]) * 1.5,
        color=ACC,
        alpha=0.85,
        edgecolor="white",
        lw=0.8,
        zorder=3,
        label="out-of-sample deciles",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("Predicted win probability", fontsize=10)
    ax.set_ylabel("Actual win rate", fontsize=10)
    ax.set_title(
        "Predicted probability reads as a real win rate\n"
        "115,841 out-of-sample runners, isotonic calibrated",
        fontsize=10.5,
        loc="left",
        pad=10,
    )
    ax.grid(color=GRID, lw=0.7, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "anaba_calibration.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("fig6 ok")


if __name__ == "__main__":
    fig_crossing()
    fig_oddsband()
    fig_racetype()
    fig_robust()
    fig_calib()
    fig_bankroll()
