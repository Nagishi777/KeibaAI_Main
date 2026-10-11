"""Walk-forward out-of-sample evaluation of the steam-move overlay model.

Protocol (no future information at any point):
  - Time-ordered folds, each 3 months of test data.
  - For fold k, the model is fit only on races strictly before the fold start.
  - The last 20% of that training window (by date) is a holdout used for
    early stopping and isotonic calibration.
  - Bet price is the 1m odds; settlement uses the confirmed payout.
  - Every threshold (EV, odds band) is fixed a priori, not tuned per fold.
"""

from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from src.backtest.anaba.features import FEATURES, FEATURES_TOP8, build

pd.set_option("display.width", 220)

PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "max_depth": 5,
    "learning_rate": 0.03,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "verbose": -1,
    "seed": 0,
    "num_threads": 8,
}

MIN_TRAIN_RACES = 2500  # ~9 months of racing before the first bet
FOLD_MONTHS = 3


def fold_bounds(dates: pd.Series):
    """Quarter starts that have enough history behind them."""
    q = dates.dt.to_period("Q").unique()
    q = sorted(q)
    return q


def run(
    dataset_path: Path,
    out_dir: Path,
    seed: int = 0,
    verbose: bool = True,
    feature_set: list[str] | None = None,
):
    feats = list(feature_set) if feature_set is not None else FEATURES
    raw = pd.read_feather(dataset_path)
    d = build(raw)
    # require the full market picture; do NOT impute
    need = [
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
    before = len(d)
    d = d.dropna(subset=need).reset_index(drop=True)
    if verbose:
        print(f"rows {before} -> {len(d)} (dropped {before - len(d)} incomplete)")
        print("races", d["race_id"].nunique())

    d["quarter"] = d["date"].dt.to_period("Q")
    quarters = sorted(d["quarter"].unique())

    preds = []
    params = dict(PARAMS)
    params["seed"] = seed

    for q in quarters:
        te = d[d["quarter"] == q]
        tr = d[d["date"] < te["date"].min()]
        if tr["race_id"].nunique() < MIN_TRAIN_RACES:
            continue
        # split train into fit / holdout by date (last 20%)
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

        # isotonic calibration on the holdout only
        ph = model.predict(hold[feats], num_iteration=model.best_iteration)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(ph, hold["won"].to_numpy())

        pt = model.predict(te[feats], num_iteration=model.best_iteration)
        out = te[
            [
                "race_id",
                "date",
                "horse_number",
                "odds_1m",
                "odds_5m",
                "final_odds",
                "payout_win",
                "payout_place",
                "won",
                "placed",
                "field_size",
                "drift_5_1",
                "inc_share_5_1",
                "pool_1m_yen",
                "mkt_1m",
                "log_pool_per_horse",
            ]
        ].copy()
        out["p_raw"] = pt
        out["p"] = iso.predict(pt)
        out["quarter"] = str(q)
        out["n_trees"] = model.best_iteration
        preds.append(out)
        if verbose:
            from sklearn.metrics import roc_auc_score

            auc = roc_auc_score(te["won"], pt)
            print(
                f"{q} train_races={tr['race_id'].nunique():5d}"
                f" test_races={te['race_id'].nunique():4d}"
                f" trees={model.best_iteration:4d} test_auc={auc:.4f}"
            )

    p = pd.concat(preds, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    p.to_feather(out_dir / f"wf_preds_seed{seed}.feather")
    if verbose:
        print(
            "\nOOS rows",
            len(p),
            "races",
            p["race_id"].nunique(),
            "range",
            p["date"].min().date(),
            p["date"].max().date(),
        )
    return p


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="穴馬戦略のウォークフォワード検証")
    ap.add_argument("--dataset", required=True, help="dataset.py が出力した Feather")
    ap.add_argument("--out-dir", required=True, help="予測の出力先ディレクトリ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--feature-set",
        choices=("all29", "top8"),
        default="all29",
        help="使う特徴量セット。top8 は削減版（回収率はほぼ同等）",
    )
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    chosen = FEATURES_TOP8 if a.feature_set == "top8" else FEATURES
    run(Path(a.dataset), Path(a.out_dir), a.seed, feature_set=chosen)
