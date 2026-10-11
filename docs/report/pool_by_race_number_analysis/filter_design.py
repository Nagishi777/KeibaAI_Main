"""フィルタ設計の材料: レース番号・プール規模別に、穴馬戦略の採用点数と
確定オッズベースの回収率（近似）を出す。

IMPORTANT (回収率は近似である):
    このリポジトリには払戻テーブル（payouts）が無い。よって的中時の払戻は
    ``horses`` CSV の確定単勝オッズ ``odds_win`` から 100円×オッズ で近似する。
    実際の払戻は100円未満の端数が切り捨てられるため、わずかに過大評価になる。
    レポート本体の数字（確定払戻で決済）とは別物として読むこと。
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = '.'

# --- 学習データと同じ経路で特徴量を作る -------------------------------------
import sys
sys.path.insert(0, ROOT)
import logging
logging.basicConfig(level=logging.ERROR)

from src.cli_common import build_config
from src.simulator.features import (
    DRIFT_COL, ODDS_COL, POOL_COL, LIVE_DECISION_SNAPSHOT,
    TRAIN_DECISION_SNAPSHOT, build_features, drop_incomplete_rows,
    normalize_late_columns, required_snapshots,
)
from src.simulator.odds_series_loader import load_odds_series
from src.simulator.predict_today import predict_proba
from src.simulator.strategy import AnabaRule, apply_rule, REASON_BET

cfg = build_config('config/config.yaml')

# holdout 期間（木の学習に使っていない 2026-01-26 以降）で評価する
raw = load_odds_series('data/processed/odds_series', '202602', '202607',
                       snapshots=required_snapshots(TRAIN_DECISION_SNAPSHOT))
w = build_features(normalize_late_columns(raw, TRAIN_DECISION_SNAPSHOT))
w = drop_incomplete_rows(w, context='分析')

# 着順と確定オッズを結合
w['date'] = pd.to_datetime(w['date'])
w['race_id_i'] = w.race_id.astype('int64')
frames = []
for f in sorted(glob.glob('data/processed/horses/*_horses.csv')):
    d = pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                    usecols=['race_id', 'horse_number', 'finish_position', 'odds_win'])
    frames.append(d)
h = pd.concat(frames, ignore_index=True).drop_duplicates(['race_id', 'horse_number'])
w = w.merge(h.rename(columns={'race_id': 'race_id_i'}),
            on=['race_id_i', 'horse_number'], how='left')
w = w[w.finish_position.notna() & (w.finish_position > 0)].copy()
w['won'] = (w.finish_position == 1).astype(int)
w['final_odds'] = pd.to_numeric(w.odds_win, errors='coerce')

p, _ = predict_proba(cfg, w, 'win/win_anaba.txt')
w['raceno'] = w.race_id.astype(str).str[-2:].astype(int)
w['pool_oku'] = w[POOL_COL] / 1e8

print(f'評価対象: {len(w):,}行 / {w.race_id.nunique():,}レース'
      f' (2026-02 〜 2026-07 / 木の学習に未使用の期間)')
print()


def roi(bets: pd.DataFrame) -> tuple:
    """近似回収率（確定単勝オッズ×100円で払戻を近似）。"""
    if not len(bets):
        return 0, float('nan'), float('nan'), float('nan')
    ret = (bets.won * bets.final_odds * 100).sum()
    stake = len(bets) * 100
    return (len(bets), bets.won.mean() * 100, ret / stake * 100,
            bets[ODDS_COL].median())


def report(title: str, rule: AnabaRule, extra_mask=None) -> None:
    out = apply_rule(w, p, rule)
    b = out[out.skip_reason == REASON_BET]
    if extra_mask is not None:
        b = b[extra_mask.reindex(b.index).fillna(False)]
    n, hit, r, od = roi(b)
    days = w.date.dt.date.nunique()
    print(f'  {title:34s} {n:4d}点  的中{hit:5.2f}%  回収{r:6.1f}%  '
          f'中央オッズ{od:5.1f}  1日{n / days:4.2f}点')


print('=== [E-1] 総投票額しきい値を変えたとき（他条件は指示どおり）===')
print('  ※回収率は確定単勝オッズによる近似（端数切捨て前）。参考値。')
for thr in [0, 3, 4, 5, 6, 7, 8, 10]:
    report(f'総投票額 >= {thr}億円', AnabaRule(min_pool_yen=thr * 1e8))
print()

print('=== [E-2] レース番号で絞る（総投票額フィルタは外す）===')
out = apply_rule(w, p, AnabaRule(min_pool_yen=0))
b_all = out[out.skip_reason == REASON_BET]
for lo in [1, 5, 7, 9, 10]:
    b = b_all[b_all.raceno >= lo]
    n, hit, r, od = roi(b)
    print(f'  R{lo}以降のみ{"":22s} {n:4d}点  的中{hit:5.2f}%  回収{r:6.1f}%  '
          f'中央オッズ{od:5.1f}')
print()

print('=== [E-3] レース番号別の採用点数と回収率（総投票額フィルタ無し）===')
t = b_all.groupby('raceno').apply(
    lambda s: pd.Series(dict(zip(['n', 'hit%', 'roi%', 'odds'], roi(s)))),
    include_groups=False)
print(t.round(2).to_string())
print()

print('=== [E-4] プール規模帯別（総投票額フィルタ無しで採用した点を分類）===')
bins = [0, 3, 5, 7, 10, 20, 1e9]
labels = ['0-3億', '3-5億', '5-7億', '7-10億', '10-20億', '20億+']
b_all = b_all.copy()
b_all['band'] = pd.cut(b_all.pool_oku, bins=bins, labels=labels)
t = b_all.groupby('band', observed=True).apply(
    lambda s: pd.Series(dict(zip(['n', 'hit%', 'roi%', 'odds'], roi(s)))),
    include_groups=False)
print(t.round(2).to_string())
