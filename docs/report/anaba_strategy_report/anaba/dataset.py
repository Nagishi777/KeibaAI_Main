"""穴馬戦略用データセット構築: 時系列オッズ + 着順 + 払戻 + レース属性を結合する。

出力は馬単位のワイド表（Feather）。``1m`` スナップショットが実質的に更新
されるのは 2023 年以降なので、既定の期間はそこから始める
（``docs/anaba_strategy_report.md`` §2）。
"""

import glob
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_START, DEFAULT_END = "202301", "202607"
SNAPS = ("60m", "30m", "10m", "5m", "1m")


def periods(start: str, end: str):
    ps = []
    pattern = str(ROOT / "data/processed/odds_series/*_odds_series_tansho.csv")
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"(\d{6})_odds_series", p)
        if m and start <= m.group(1) <= end:
            ps.append(m.group(1))
    return ps


def load_series(ps):
    need = ["race_id", "date", "post_datetime", "umaban"]
    for s in SNAPS:
        need += ["odds_win_" + s, "hyosu_total_" + s]
    frames = []
    for per in ps:
        path = ROOT / ("data/processed/odds_series/" + per + "_odds_series_tansho.csv")
        d = pd.read_csv(path, encoding="utf-8-sig", usecols=need, low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"umaban": "horse_number"})
    df["race_id"] = df["race_id"].astype(str)
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").astype(
        "Int64"
    )
    df["date"] = pd.to_datetime(df["date"])
    for s in SNAPS:
        o = pd.to_numeric(df["odds_win_" + s], errors="coerce")
        df["odds_" + s] = o.where(o > 0)
        p = pd.to_numeric(df["hyosu_total_" + s], errors="coerce")
        df["pool_" + s] = p.where(p > 0)
        df = df.drop(columns=["odds_win_" + s, "hyosu_total_" + s])
    return df


def load_results(ps):
    cols = [
        "race_id",
        "finish_position",
        "horse_number",
        "odds_win",
        "popularity",
        "sex_age",
        "impost",
        "horse_weight",
        "horse_weight_change",
        "jockey_id",
        "trainer_id",
        "horse_id",
    ]
    frames = []
    for per in ps:
        path = ROOT / ("data/processed/horses/" + per + "_horses.csv")
        d = pd.read_csv(path, encoding="utf-8-sig", usecols=cols, low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = df["race_id"].astype(str)
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce").astype(
        "Int64"
    )
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")
    df["final_odds"] = pd.to_numeric(df["odds_win"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    return df.drop(columns=["odds_win"])


def load_races(ps):
    frames = []
    for per in ps:
        path = ROOT / ("data/processed/races/" + per + "_races.csv")
        d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = df["race_id"].astype(str)
    df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
    keep = [
        "race_id",
        "distance",
        "track_type",
        "weather",
        "track_condition",
        "race_class",
        "racecourse",
        "race_name",
    ]
    return df[keep]


def load_payouts(ps):
    cols = {
        "race_id",
        "tan",
        "tan_amount",
        "fuku_1",
        "fuku_1_amount",
        "fuku_2",
        "fuku_2_amount",
        "fuku_3",
        "fuku_3_amount",
        "fuku_4",
        "fuku_4_amount",
    }
    frames = []
    for per in ps:
        path = ROOT / ("data/processed/payouts/" + per + "_payouts.csv")
        d = pd.read_csv(
            path, encoding="utf-8-sig", usecols=lambda c: c in cols, low_memory=False
        )
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["race_id"] = df["race_id"].astype(str)
    return df


def main(out_path: Path, start: str = DEFAULT_START, end: str = DEFAULT_END):
    ps = periods(start, end)
    print("periods:", len(ps), ps[0], "..", ps[-1])
    ser = load_series(ps)
    print("series", ser.shape, ser["race_id"].nunique())
    res = load_results(ps)
    print("results", res.shape, res["race_id"].nunique())
    rac = load_races(ps)
    pay = load_payouts(ps)

    df = ser.merge(res, on=["race_id", "horse_number"], how="inner")
    print("after results merge", df.shape, df["race_id"].nunique())
    df = df.merge(rac, on="race_id", how="left")

    pw = pay[["race_id", "tan", "tan_amount"]].copy()
    pw["tan"] = pd.to_numeric(pw["tan"], errors="coerce")
    pw["tan_amount"] = pd.to_numeric(pw["tan_amount"], errors="coerce")
    pw = pw.dropna(subset=["tan"])
    pw["horse_number"] = pw["tan"].astype("int64")
    pw = pw[["race_id", "horse_number", "tan_amount"]].rename(
        columns={"tan_amount": "payout_win"}
    )
    pw = pw.drop_duplicates(["race_id", "horse_number"])
    df = df.merge(pw, on=["race_id", "horse_number"], how="left")
    df["payout_win"] = df["payout_win"].fillna(0.0)

    longs = []
    for i in (1, 2, 3, 4):
        a = "fuku_" + str(i)
        b = "fuku_" + str(i) + "_amount"
        if a not in pay.columns:
            continue
        t = pay[["race_id", a, b]].copy()
        t[a] = pd.to_numeric(t[a], errors="coerce")
        t[b] = pd.to_numeric(t[b], errors="coerce")
        t = t.dropna(subset=[a])
        t["horse_number"] = t[a].astype("int64")
        longs.append(
            t[["race_id", "horse_number", b]].rename(columns={b: "payout_place"})
        )
    pl = pd.concat(longs, ignore_index=True).drop_duplicates(
        ["race_id", "horse_number"]
    )
    df = df.merge(pl, on=["race_id", "horse_number"], how="left")
    df["payout_place"] = df["payout_place"].fillna(0.0)

    df["field_size"] = df.groupby("race_id")["horse_number"].transform("size")
    df["won"] = (df["finish_position"] == 1).astype("int8")
    df["placed"] = (df["payout_place"] > 0).astype("int8")

    df = df.sort_values(["date", "race_id", "horse_number"]).reset_index(drop=True)
    df.to_feather(out_path)
    print("saved", out_path, df.shape)
    print("date range", df["date"].min(), df["date"].max())
    print(
        "win rate",
        round(df["won"].mean(), 5),
        "place rate",
        round(df["placed"].mean(), 5),
    )
    vc = (
        df.groupby("race_id")["payout_win"]
        .apply(lambda s: (s > 0).sum())
        .value_counts()
    )
    print("winners per race:", dict(vc.head()))


def _parse_args():
    import argparse

    ap = argparse.ArgumentParser(description="穴馬戦略用データセットを構築する")
    ap.add_argument("--out", required=True, help="出力 Feather パス")
    ap.add_argument("--start-period", default=DEFAULT_START)
    ap.add_argument("--end-period", default=DEFAULT_END)
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    main(Path(a.out), a.start_period, a.end_period)
