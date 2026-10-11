"""特徴量を削減できるかを検証する（アブレーション）。

やること:
  1. LightGBM の gain importance を学習期間だけで算出する
  2. 相関の高い列ペアを洗い出す
  3. 特徴量セットを差し替えてウォークフォワード検証し、回収率を比較する
  4. ALL29 と TOP8 の差が有意かを、レース単位の対応ブートストラップで判定する

結果は ``docs/anaba_feature_ablation_report.md`` にまとめている。

IMPORTANT (比較の公平性):
    特徴量セット以外（フォールド分割・ハイパーパラメータ・校正・購入ルール・決済）は
    本番と完全に揃える。列数だけを変えた比較になるようにしている。
"""

import argparse
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from src.backtest.anaba.features import FEATURES, FEATURES_TOP8, build
from src.backtest.anaba.walkforward import MIN_TRAIN_RACES, PARAMS

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = Path(os.environ.get("ANABA_WORK_DIR", ROOT / "output" / "anaba"))

# 学習期間の終わり（本番の検証と揃える）
TRAIN_END = pd.Timestamp("2025-07-27")

# 採用ルール
MIN_EV = 1.20
ODDS_MIN, ODDS_MAX = 10.0, 150.0
MAX_DRIFT = -0.05

NEED = [
    "odds_1m",
    "odds_5m",
    "odds_10m",
    "odds_30m",
    "odds_60m",
    "pool_1m",
    "pool_5m",
    "drift_5_1",
    "inc_share_5_1",
]

# 相関で重複する列を落としたセット（gain の高い方を残す）
DEDUP = [
    f
    for f in FEATURES
    if f
    not in (
        "is_dirt",  # is_turf と r=1.00
        "log_odds_5m",  # log_odds_1m と r=0.99
        "mkt_rank_1m",  # log_odds_1m と r=0.94
        "hhi_1m",  # fav_strength と r=0.95
        "log_pool_per_horse",  # log_pool_1m と r=0.94
        "inc_ratio",  # drift_5_1 と r=0.94
    )
]

# 市場列だけ（レース属性・馬体重を落とす）
MARKET_ONLY = [
    f
    for f in FEATURES
    if f
    not in (
        "field_size",
        "distance",
        "is_turf",
        "is_dirt",
        "age",
        "impost",
        "horse_weight",
        "horse_weight_change",
    )
]

TOP4 = FEATURES_TOP8[:4]
TOP12 = FEATURES_TOP8 + [
    "drift_60_1",
    "fav_strength",
    "horse_weight_change",
    "drift_10_5",
]

SETS: dict[str, list[str]] = {
    "ALL29": FEATURES,
    "DEDUP23": DEDUP,
    "MARKET21": MARKET_ONLY,
    "TOP12": TOP12,
    "TOP8": FEATURES_TOP8,
    "TOP4": TOP4,
    "TOP2": ["inc_share_5_1", "log_odds_1m"],
}


def load(work_dir: Path) -> pd.DataFrame:
    """base.feather を読み、特徴量を作って欠損行を落とす。"""
    path = work_dir / "base.feather"
    if not path.is_file():
        raise FileNotFoundError(
            f"データセットがありません: {path}。"
            " 先に src.backtest.anaba.dataset を実行してください。"
        )
    d = build(pd.read_feather(path))
    d = d.dropna(subset=NEED).reset_index(drop=True)
    d["quarter"] = d["date"].dt.to_period("Q")
    return d


def importance(d: pd.DataFrame, n_seeds: int = 3) -> pd.DataFrame:
    """学習期間だけで gain importance を出す（検証期間は使わない）。"""
    tr = d[d["date"] <= TRAIN_END]
    gain = np.zeros(len(FEATURES))
    split = np.zeros(len(FEATURES))
    params = dict(PARAMS)
    for seed in range(n_seeds):
        params["seed"] = seed
        ds = lgb.Dataset(tr[FEATURES], label=tr["won"])
        m = lgb.train(params, ds, num_boost_round=200)
        gain += m.feature_importance("gain")
        split += m.feature_importance("split")
    imp = pd.DataFrame(
        {
            "feature": FEATURES,
            "gain": gain / n_seeds,
            "split": split / n_seeds,
        }
    )
    imp["gain_pct"] = imp["gain"] / imp["gain"].sum() * 100
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["cum_pct"] = imp["gain_pct"].cumsum()
    return imp


def correlated_pairs(d: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """相関の高い列ペアを列挙する。"""
    c = d[FEATURES].corr().abs()
    rows = []
    for i, a in enumerate(FEATURES):
        for b in FEATURES[i + 1 :]:
            r = float(c.loc[a, b])
            if r >= threshold:
                rows.append({"a": a, "b": b, "corr": r})
    return pd.DataFrame(rows).sort_values("corr", ascending=False)


def run_walkforward(d: pd.DataFrame, feats: list[str], seed: int) -> pd.DataFrame:
    """特徴量セットを指定してウォークフォワード予測を作る。"""
    params = dict(PARAMS)
    params["seed"] = seed
    preds = []
    for q in sorted(d["quarter"].unique()):
        te = d[d["quarter"] == q]
        tr = d[d["date"] < te["date"].min()]
        if tr["race_id"].nunique() < MIN_TRAIN_RACES:
            continue
        cut = tr["date"].quantile(0.80)
        fit = tr[tr["date"] <= cut]
        hold = tr[tr["date"] > cut]
        if len(hold) < 5000:
            continue
        dtrain = lgb.Dataset(fit[feats], label=fit["won"])
        dvalid = lgb.Dataset(hold[feats], label=hold["won"], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        ph = model.predict(hold[feats], num_iteration=model.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(ph, hold["won"].to_numpy())
        pt = model.predict(te[feats], num_iteration=model.best_iteration)
        out = te[
            ["race_id", "date", "odds_1m", "drift_5_1", "payout_win", "won"]
        ].copy()
        out["p"] = iso.predict(pt)
        out["auc_raw"] = roc_auc_score(te["won"], pt)
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def bets_of(p: pd.DataFrame) -> pd.DataFrame:
    """採用ルールで購入対象だけを残す。"""
    p = p.copy()
    p["ev"] = p["p"] * p["odds_1m"]
    mask = (
        (p["ev"] >= MIN_EV)
        & p["odds_1m"].between(ODDS_MIN, ODDS_MAX)
        & (p["drift_5_1"] <= MAX_DRIFT)
    )
    return p[mask]


def evaluate(p: pd.DataFrame) -> dict:
    """回収率・的中率をまとめる。"""
    b = bets_of(p)
    if len(b) == 0:
        return {"bets": 0, "hits": 0, "hit": np.nan, "roi": np.nan, "auc": np.nan}
    return {
        "bets": len(b),
        "hits": int(b["won"].sum()),
        "hit": b["won"].mean(),
        "roi": (b["payout_win"] / 100.0).sum() / len(b),
        "auc": (
            p.groupby("date")["auc_raw"].first().mean()
            if "auc_raw" in p.columns
            else np.nan
        ),
    }


def ensemble(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """複数シードの予測確率を平均する（本番と同じ使い方）。"""
    base = preds[0][
        ["race_id", "date", "odds_1m", "drift_5_1", "payout_win", "won"]
    ].copy()
    base["p"] = np.column_stack([x["p"].to_numpy() for x in preds]).mean(axis=1)
    return base


def paired_diff_ci(
    a: pd.DataFrame, b: pd.DataFrame, n: int = 5000, seed: int = 20260103
) -> tuple[float, float, float]:
    """2セットの回収率の差を、レース単位の対応ブートストラップで評価する。

    Returns:
        tuple: (観測差, CI下限, CI上限)
    """
    rng = np.random.default_rng(seed)
    races = sorted(set(a["race_id"]) | set(b["race_id"]))
    idx = {r: i for i, r in enumerate(races)}
    n_races = len(races)

    def agg(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        st = np.zeros(n_races)
        rt = np.zeros(n_races)
        codes = x["race_id"].map(idx).to_numpy()
        np.add.at(st, codes, 100.0)
        np.add.at(rt, codes, x["payout_win"].to_numpy(float))
        return st, rt

    a_st, a_rt = agg(a)
    b_st, b_rt = agg(b)
    obs = a_rt.sum() / a_st.sum() - b_rt.sum() / b_st.sum()
    draws = rng.integers(0, n_races, size=(n, n_races))
    diffs = np.empty(n)
    for i, row in enumerate(draws):
        ra = a_rt[row].sum() / max(a_st[row].sum(), 1e-9)
        rb = b_rt[row].sum() / max(b_st[row].sum(), 1e-9)
        diffs[i] = ra - rb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="穴馬戦略の特徴量アブレーション")
    ap.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK),
        help="base.feather のあるディレクトリ",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "anaba"),
        help="重要度CSVの出力先",
    )
    ap.add_argument("--seeds", type=int, default=8, help="評価に使うシード数")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load(work)
    print(f"rows {len(d)} races {d['race_id'].nunique()}")

    print("\n=== gain importance（学習期間のみ・3シード平均）===")
    imp = importance(d)
    print(imp.round(2).to_string(index=False))
    imp_path = out_dir / "anaba_feature_importance.csv"
    imp.to_csv(imp_path, index=False, encoding="utf-8-sig")
    print(f"saved {imp_path}")
    print(f"下位15列の gain 合計 = {imp.tail(15)['gain_pct'].sum():.2f}%")

    print("\n=== 相関の高いペア (|r| >= 0.85) ===")
    print(correlated_pairs(d).round(4).to_string(index=False))

    print(f"\n=== セット別の回収率（{args.seeds}シード）===")
    rows = []
    store: dict[str, list[pd.DataFrame]] = {}
    for name, feats in SETS.items():
        store[name] = []
        for seed in range(args.seeds):
            p = run_walkforward(d, feats, seed)
            store[name].append(p)
            r = evaluate(p)
            r.update({"set": name, "n_feat": len(feats), "seed": seed})
            rows.append(r)
        print(f"{name} done", flush=True)

    res = pd.DataFrame(rows)
    g = (
        res.groupby(["set", "n_feat"])
        .agg(
            bets=("bets", "mean"),
            hit=("hit", "mean"),
            roi_mean=("roi", "mean"),
            roi_std=("roi", "std"),
            auc=("auc", "mean"),
        )
        .reset_index()
        .sort_values("n_feat", ascending=False)
    )
    print(g.round(4).to_string(index=False))

    print("\n=== アンサンブル（本番と同じ使い方）===")
    erows = []
    ens: dict[str, pd.DataFrame] = {}
    for name, lst in store.items():
        ens[name] = ensemble(lst)
        r = evaluate(ens[name])
        r.update({"set": name, "n_feat": len(SETS[name])})
        erows.append(r)
    e = pd.DataFrame(erows).sort_values("n_feat", ascending=False)
    print(
        e[["set", "n_feat", "bets", "hits", "hit", "roi"]]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== ALL29 と TOP8 の差（対応ブートストラップ）===")
    obs, lo, hi = paired_diff_ci(bets_of(ens["ALL29"]), bets_of(ens["TOP8"]))
    print(f"観測差 (ALL29 - TOP8) = {obs:+.4f}")
    print(f"95%CI = [{lo:+.4f}, {hi:+.4f}]")
    print(f"0 を含む: {lo <= 0 <= hi}（含むなら差は有意でない）")


if __name__ == "__main__":
    main()
