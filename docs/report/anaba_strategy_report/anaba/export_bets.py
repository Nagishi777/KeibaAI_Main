"""穴馬戦略でベットした全レースの明細を CSV に書き出す。

出力するファイル:
  - anaba_bets_detail.csv         flat・kelly 両方の金額列を持つ全体明細
  - anaba_bets_monthly.csv        同上の月次集計
  - anaba_bets_flat.csv           均等賭け（flat）だけの明細
  - anaba_bets_flat_monthly.csv   flat の月次集計
  - anaba_bets_kelly.csv          fractional Kelly だけの明細
  - anaba_bets_kelly_monthly.csv  kelly の月次集計

明細に含む情報:
  - いつ・どのレース・どの馬に賭けたか（日付・レースID・レース名・馬名）
  - ベットの根拠（予測確率・1mオッズ・期待値・オッズの下落率・資金流入シェア）
  - ベット額・払戻・損益・累積資金残高（各方式ごと）
  - 的中有無

IMPORTANT (順序):
    資金推移は「日付 → 発走時刻 → レースID」の順に確定させる。
    同一日内の並び順で最大ドローダウンと連続不的中が変わるため、
    実際に賭ける順序（＝発走順）を再現する。

IMPORTANT (flat と kelly を比較する際の注意):
    kelly は複利なので、資金残高は当日の直前レースの結果に依存する。
    flat の資金推移とは前提が異なる（flat は常に1点1,000円固定）ため、
    ``stake`` ``return`` ``pnl`` ``bankroll`` の値そのものを直接比較するのではなく、
    回収率（ROI）や資金の伸び方の「形」を比べるのが適切である。
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.anaba.strategy import AnabaRule, kelly_stake, select

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WORK = Path(os.environ.get("ANABA_WORK_DIR", ROOT / "output" / "anaba"))
SEEDS = (0, 1, 2, 3, 4)

# 均等賭けの1点あたり金額（円）
FLAT_STAKE = 1000.0
# Kelly の初期資金（円）
KELLY_BANKROLL = 300000.0
KELLY_FRACTION = 0.10


def load_ensemble(work_dir: Path) -> pd.DataFrame:
    """5シードの予測を平均し、レース属性を結合する。

    Args:
        work_dir: ``wf_preds_seed*.feather`` と ``base.feather`` のある場所

    Returns:
        pd.DataFrame: 予測確率 ``p`` を持つ全出走馬の表

    Raises:
        FileNotFoundError: 必要なファイルが無い場合
    """
    frames = []
    for s in SEEDS:
        path = work_dir / f"wf_preds_seed{s}.feather"
        if not path.is_file():
            raise FileNotFoundError(
                f"予測ファイルがありません: {path}。"
                " 先に src.backtest.anaba.walkforward を実行してください。"
            )
        q = pd.read_feather(path)[["race_id", "horse_number", "p"]]
        frames.append(
            q.rename(columns={"p": f"p{s}"}).set_index(["race_id", "horse_number"])
        )
    merged = pd.concat(frames, axis=1)
    merged["p"] = merged[[f"p{s}" for s in SEEDS]].mean(axis=1)

    base_pred = pd.read_feather(work_dir / "wf_preds_seed0.feather").drop(columns=["p"])
    out = (
        base_pred.set_index(["race_id", "horse_number"])
        .join(merged[["p"]])
        .reset_index()
    )

    base = pd.read_feather(work_dir / "base.feather")
    keep = [
        "race_id",
        "horse_number",
        "post_datetime",
        "finish_position",
        "popularity",
        "race_name",
        "race_class",
        "racecourse",
        "distance",
        "track_type",
        "track_condition",
    ]
    return out.merge(base[keep], on=["race_id", "horse_number"], how="left")


def attach_names(df: pd.DataFrame) -> pd.DataFrame:
    """馬名・騎手名を月別 CSV から結合する。"""
    # 月別ファイルは開催日基準なので date から年月を決める
    periods = sorted(df["date"].dt.strftime("%Y%m").unique())
    frames = []
    for per in periods:
        path = ROOT / f"data/processed/horses/{per}_horses.csv"
        if not path.is_file():
            continue
        d = pd.read_csv(
            path,
            encoding="utf-8-sig",
            usecols=["race_id", "horse_number", "horse_name", "jockey"],
            low_memory=False,
        )
        frames.append(d)
    if not frames:
        raise FileNotFoundError("馬名を引く horses CSV が見つかりませんでした。")
    names = pd.concat(frames, ignore_index=True)
    names["race_id"] = names["race_id"].astype(str)
    names["horse_number"] = pd.to_numeric(names["horse_number"], errors="coerce")
    df = df.copy()
    df["horse_number"] = pd.to_numeric(df["horse_number"], errors="coerce")
    return df.merge(names, on=["race_id", "horse_number"], how="left")


def build(work_dir: Path, rule: AnabaRule | None = None) -> pd.DataFrame:
    """ベット明細表を作る。

    Args:
        work_dir: 予測ファイルのある場所
        rule: 選別ルール。省略時は採用条件

    Returns:
        pd.DataFrame: 1行1ベットの明細（発走順）
    """
    rule = rule or AnabaRule()
    ens = load_ensemble(work_dir)
    bets = select(ens, rule)
    bets = attach_names(bets)

    # --- 発走順に並べる（実際に賭ける順序）---
    bets["post_time"] = bets["post_datetime"].astype(str).str.slice(0, 5)
    bets = bets.sort_values(
        ["date", "post_time", "race_id", "ev"], ascending=[True, True, True, False]
    ).reset_index(drop=True)

    # --- レース番号は race_id の末尾2桁 ---
    bets["race_no"] = bets["race_id"].astype(str).str[-2:]

    # --- ベット額 ---
    bets["stake_flat"] = FLAT_STAKE
    bets["return_flat"] = bets["payout_win"] / 100.0 * FLAT_STAKE
    bets["pnl_flat"] = bets["return_flat"] - bets["stake_flat"]
    bets["bankroll_flat"] = KELLY_BANKROLL + bets["pnl_flat"].cumsum()

    # --- Kelly は複利なのでレース単位で逐次計算する ---
    bank = KELLY_BANKROLL
    stakes, returns, banks = [], [], []
    for _, grp in bets.groupby(["date", "post_time", "race_id"], sort=False):
        st = kelly_stake(
            grp["p"].to_numpy(dtype="float64"),
            grp["odds_1m"].to_numpy(dtype="float64"),
            bank,
            fraction=KELLY_FRACTION,
        )
        # 同一レース内で資金を超えないよう按分する
        total = float(st.sum())
        if total > bank:
            st = np.floor(st * (bank / total) / 100.0) * 100.0
        ret = st * grp["payout_win"].to_numpy(dtype="float64") / 100.0
        bank += float(ret.sum() - st.sum())
        stakes.extend(st.tolist())
        returns.extend(ret.tolist())
        banks.extend([bank] * len(grp))
    bets["stake_kelly"] = stakes
    bets["return_kelly"] = returns
    bets["pnl_kelly"] = bets["return_kelly"] - bets["stake_kelly"]
    bets["bankroll_kelly"] = banks

    # --- 根拠の可読化 ---
    bets["odds_drop_pct"] = (bets["odds_1m"] / bets["odds_5m"] - 1.0) * 100.0
    bets["implied_prob"] = 0.8 / bets["odds_1m"]
    bets["edge_ratio"] = bets["p"] / bets["implied_prob"]
    bets["hit"] = np.where(bets["won"] == 1, "的中", "不的中")
    bets["pool_1m_oku"] = bets["pool_1m_yen"] / 1e8

    cols = [
        # いつ・どこ
        "date",
        "post_time",
        "race_id",
        "race_no",
        "racecourse",
        "race_name",
        "race_class",
        "distance",
        "track_type",
        "track_condition",
        "field_size",
        # 誰に
        "horse_number",
        "horse_name",
        "jockey",
        "popularity",
        # 根拠
        "odds_5m",
        "odds_1m",
        "odds_drop_pct",
        "drift_5_1",
        "inc_share_5_1",
        "pool_1m_oku",
        "p",
        "implied_prob",
        "edge_ratio",
        "ev",
        "ev_rank",
        # 結果
        "finish_position",
        "hit",
        "won",
        "final_odds",
        "payout_win",
        # 金額と資金推移
        "stake_flat",
        "return_flat",
        "pnl_flat",
        "bankroll_flat",
        "stake_kelly",
        "return_kelly",
        "pnl_kelly",
        "bankroll_kelly",
    ]
    out = bets[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    round_map = {
        "odds_drop_pct": 2,
        "drift_5_1": 4,
        "inc_share_5_1": 5,
        "pool_1m_oku": 2,
        "p": 5,
        "implied_prob": 5,
        "edge_ratio": 3,
        "ev": 4,
        "stake_kelly": 0,
        "return_kelly": 0,
        "pnl_kelly": 0,
        "bankroll_kelly": 0,
        "bankroll_flat": 0,
        "return_flat": 0,
        "pnl_flat": 0,
    }
    for c, nd in round_map.items():
        out[c] = out[c].round(nd)
    return out


def summarize(bets: pd.DataFrame) -> pd.DataFrame:
    """月次の集計表を作る（flat・kelly 両方の列を含む後方互換版）。"""
    b = bets.copy()
    b["month"] = b["date"].str.slice(0, 7)
    g = b.groupby("month").apply(
        lambda x: pd.Series(
            {
                "bets": len(x),
                "races": x["race_id"].nunique(),
                "hits": int((x["won"] == 1).sum()),
                "hit_rate": round((x["won"] == 1).mean(), 4),
                "mean_odds": round(x["odds_1m"].mean(), 2),
                "stake_flat": x["stake_flat"].sum(),
                "return_flat": x["return_flat"].sum(),
                "pnl_flat": x["pnl_flat"].sum(),
                "roi_flat": round(x["return_flat"].sum() / x["stake_flat"].sum(), 4),
                "bankroll_flat_end": x["bankroll_flat"].iloc[-1],
                "bankroll_kelly_end": x["bankroll_kelly"].iloc[-1],
            }
        ),
        include_groups=False,
    )
    return g.reset_index()


# 各賭け方式で明細・月次それぞれに残す列。
# 「いつ・誰に・根拠・結果」は共通で、金額系だけを賭け方式で絞り込む。
_COMMON_COLS = [
    "date",
    "post_time",
    "race_id",
    "race_no",
    "racecourse",
    "race_name",
    "race_class",
    "distance",
    "track_type",
    "track_condition",
    "field_size",
    "horse_number",
    "horse_name",
    "jockey",
    "popularity",
    "odds_5m",
    "odds_1m",
    "odds_drop_pct",
    "drift_5_1",
    "inc_share_5_1",
    "pool_1m_oku",
    "p",
    "implied_prob",
    "edge_ratio",
    "ev",
    "ev_rank",
    "finish_position",
    "hit",
    "won",
    "final_odds",
    "payout_win",
]
_METHOD_COLS = {
    "flat": ["stake_flat", "return_flat", "pnl_flat", "bankroll_flat"],
    "kelly": ["stake_kelly", "return_kelly", "pnl_kelly", "bankroll_kelly"],
}


def split_by_method(bets: pd.DataFrame, method: str) -> pd.DataFrame:
    """明細表から指定した賭け方式（flat / kelly）だけの表を切り出す。

    列名から ``_flat`` / ``_kelly`` の接尾辞を外し、両方式で同じ列名
    （``stake`` ``return`` ``pnl`` ``bankroll``）にそろえる。
    そのため flat 用と kelly 用を並べて突き合わせやすい。

    Args:
        bets: ``build()`` が返す明細表
        method: ``"flat"`` または ``"kelly"``

    Returns:
        pd.DataFrame: 指定方式の金額列だけを持つ明細表

    Raises:
        ValueError: method が未知の場合
    """
    if method not in _METHOD_COLS:
        raise ValueError(f"method は 'flat' か 'kelly' を指定してください: {method}")
    rename = {
        f"stake_{method}": "stake",
        f"return_{method}": "return",
        f"pnl_{method}": "pnl",
        f"bankroll_{method}": "bankroll",
    }
    cols = _COMMON_COLS + _METHOD_COLS[method]
    out = bets[cols].rename(columns=rename)
    return out.reset_index(drop=True)


def summarize_method(bets: pd.DataFrame, method: str) -> pd.DataFrame:
    """指定した賭け方式だけの月次集計表を作る。

    Args:
        bets: ``build()`` が返す明細表（分割前）
        method: ``"flat"`` または ``"kelly"``

    Returns:
        pd.DataFrame: 月ごとの点数・的中・回収率・月末残高
    """
    single = split_by_method(bets, method)
    b = single.copy()
    b["month"] = b["date"].str.slice(0, 7)
    g = b.groupby("month").apply(
        lambda x: pd.Series(
            {
                "bets": len(x),
                "races": x["race_id"].nunique(),
                "hits": int((x["won"] == 1).sum()),
                "hit_rate": round((x["won"] == 1).mean(), 4),
                "mean_odds": round(x["odds_1m"].mean(), 2),
                "stake": x["stake"].sum(),
                "return": x["return"].sum(),
                "pnl": x["pnl"].sum(),
                "roi": round(x["return"].sum() / x["stake"].sum(), 4),
                "bankroll_end": x["bankroll"].iloc[-1],
            }
        ),
        include_groups=False,
    )
    return g.reset_index()


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="穴馬戦略のベット明細を CSV に出力する")
    ap.add_argument(
        "--work-dir",
        default=str(DEFAULT_WORK),
        help="wf_preds_seed*.feather と base.feather のあるディレクトリ",
    )
    ap.add_argument(
        "--out-dir",
        default=str(ROOT / "output" / "anaba"),
        help="CSV の出力先",
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bets = build(work)
    detail = out_dir / "anaba_bets_detail.csv"
    bets.to_csv(detail, index=False, encoding="utf-8-sig")

    monthly = summarize(bets)
    summary = out_dir / "anaba_bets_monthly.csv"
    monthly.to_csv(summary, index=False, encoding="utf-8-sig")

    # --- flat / kelly をそれぞれ別 CSV に分ける ---
    method_paths: dict[str, tuple[Path, Path]] = {}
    for method in ("flat", "kelly"):
        m_detail = split_by_method(bets, method)
        m_detail_path = out_dir / f"anaba_bets_{method}.csv"
        m_detail.to_csv(m_detail_path, index=False, encoding="utf-8-sig")

        m_monthly = summarize_method(bets, method)
        m_monthly_path = out_dir / f"anaba_bets_{method}_monthly.csv"
        m_monthly.to_csv(m_monthly_path, index=False, encoding="utf-8-sig")
        method_paths[method] = (m_detail_path, m_monthly_path)

    n = len(bets)
    hits = int((bets["won"] == 1).sum())
    roi = bets["return_flat"].sum() / bets["stake_flat"].sum()
    print(f"明細: {detail}  ({n} 行)")
    print(f"月次: {summary}  ({len(monthly)} 行)")
    print(f"ベット {n} 点 / 的中 {hits} 点 ({hits / n:.2%}) / 回収率 {roi:.2%}")
    print(
        f"均等賭け 最終資金 {bets['bankroll_flat'].iloc[-1]:,.0f} 円"
        f" / Kelly 最終資金 {bets['bankroll_kelly'].iloc[-1]:,.0f} 円"
    )
    for method, (d_path, s_path) in method_paths.items():
        print(f"{method}: 明細 {d_path}  月次 {s_path}")


if __name__ == "__main__":
    main()
