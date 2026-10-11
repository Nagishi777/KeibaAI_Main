"""[E] の回収率差が本物かノイズかを確認する。

n=40〜97 の小さなセルで回収率を比べているので、ブートストラップで
信頼区間を出し、「差があると言えるのか」を確かめる。
"""
import glob
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
logging.basicConfig(level=logging.ERROR)

from src.cli_common import build_config
from src.simulator.features import (
    ODDS_COL, POOL_COL, TRAIN_DECISION_SNAPSHOT, build_features,
    drop_incomplete_rows, normalize_late_columns, required_snapshots,
)
from src.simulator.odds_series_loader import load_odds_series
from src.simulator.predict_today import predict_proba
from src.simulator.strategy import AnabaRule, apply_rule, REASON_BET

cfg = build_config('config/config.yaml')
raw = load_odds_series('data/processed/odds_series', '202602', '202607',
                       snapshots=required_snapshots(TRAIN_DECISION_SNAPSHOT))
w = drop_incomplete_rows(
    build_features(normalize_late_columns(raw, TRAIN_DECISION_SNAPSHOT)),
    context='分析')
w['date'] = pd.to_datetime(w['date'])
w['race_id_i'] = w.race_id.astype('int64')
frames = [pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                      usecols=['race_id', 'horse_number', 'finish_position', 'odds_win'])
          for f in sorted(glob.glob('data/processed/horses/*_horses.csv'))]
h = pd.concat(frames, ignore_index=True).drop_duplicates(['race_id', 'horse_number'])
w = w.merge(h.rename(columns={'race_id': 'race_id_i'}),
            on=['race_id_i', 'horse_number'], how='left')
w = w[w.finish_position.notna() & (w.finish_position > 0)].copy()
w['won'] = (w.finish_position == 1).astype(int)
w['final_odds'] = pd.to_numeric(w.odds_win, errors='coerce')
p, _ = predict_proba(cfg, w, 'win/win_anaba.txt')
w['raceno'] = w.race_id.astype(str).str[-2:].astype(int)
w['pool_oku'] = w[POOL_COL] / 1e8

out = apply_rule(w, p, AnabaRule(min_pool_yen=0))
b = out[out.skip_reason == REASON_BET].copy()
rng = np.random.default_rng(0)


def boot_roi(bets, n_boot=4000):
    """レース単位のブートストラップで回収率の95%CIを出す。"""
    if len(bets) < 5:
        return (float('nan'),) * 3
    pnl = (bets.won * bets.final_odds * 100).to_numpy()
    point = pnl.sum() / (len(bets) * 100) * 100
    idx = np.arange(len(bets))
    sims = [
        pnl[s].sum() / (len(s) * 100) * 100
        for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n_boot))
    ]
    return point, np.percentile(sims, 2.5), np.percentile(sims, 97.5)


print('=== [F-1] 総投票額しきい値別の回収率と95%信頼区間 ===')
print('  ※払戻は確定単勝オッズ×100円の近似。端数切捨て前なので上振れ。')
print(f'  {"しきい値":>12s} {"点数":>5s} {"回収率":>8s} {"95%CI":>20s}')
for thr in [0, 4, 5, 6, 7, 8, 10]:
    s = b[b.pool_oku >= thr]
    pt, lo, hi = boot_roi(s)
    flag = ' ← 下側が100%超' if lo > 100 else ''
    print(f'  {thr:>9}億円 {len(s):5d} {pt:7.1f}% [{lo:6.1f}, {hi:6.1f}]{flag}')
print()

print('=== [F-2] レース番号で絞ったとき ===')
for lo_r in [1, 5, 7, 9, 10]:
    s = b[b.raceno >= lo_r]
    pt, lo, hi = boot_roi(s)
    flag = ' ← 下側が100%超' if lo > 100 else ''
    print(f'  R{lo_r:>2}以降   {len(s):5d} {pt:7.1f}% [{lo:6.1f}, {hi:6.1f}]{flag}')
print()

print('=== [F-3] 前半 vs 後半（レース番号の効果だけを見る）===')
for label, m in [('R1-6 (前半)', b.raceno <= 6),
                 ('R7-12 (後半)', b.raceno >= 7),
                 ('R11 のみ', b.raceno == 11)]:
    s = b[m]
    pt, lo, hi = boot_roi(s)
    print(f'  {label:14s} {len(s):5d} {pt:7.1f}% [{lo:6.1f}, {hi:6.1f}]')
print()

print('=== [F-4] 総投票額とレース番号はどちらが効くのか（2軸クロス）===')
print('  行=レース番号帯 / 列=総投票額帯、各セル「点数/回収率%」')
b['rband'] = np.where(b.raceno <= 6, 'R1-6', 'R7-12')
b['pband'] = pd.cut(b.pool_oku, [0, 5, 7, 1e9], labels=['<5億', '5-7億', '7億+'])
piv = b.pivot_table(index='rband', columns='pband', observed=True,
                    values='won', aggfunc='size').fillna(0).astype(int)
roi_piv = b.pivot_table(
    index='rband', columns='pband', observed=True, values='won',
    aggfunc=lambda s: np.nan)
for r in piv.index:
    cells = []
    for c in piv.columns:
        s = b[(b.rband == r) & (b.pband == c)]
        cells.append(f'{len(s):4d}点/{(s.won * s.final_odds * 100).sum() / max(len(s), 1) :6.1f}%'
                     if len(s) else '     --      ')
    print(f'  {r:7s} ' + ' | '.join(cells))
print()

print('=== [F-5] 総投票額を「その日・その競馬場の中での相対値」にした場合 ===')
# 絶対額ではなく相対順位なら、速報値のスケール問題を回避できる可能性がある
b['pool_rank_in_day'] = b.groupby([b.date.dt.date, b.race_id.str[4:6]])['pool_oku'] \
                         .rank(pct=True)
for lo_q in [0.0, 0.3, 0.5, 0.7]:
    s = b[b.pool_rank_in_day >= lo_q]
    pt, lo, hi = boot_roi(s)
    flag = ' ← 下側が100%超' if lo > 100 else ''
    print(f'  日内上位{(1 - lo_q) * 100:3.0f}%  {len(s):5d} {pt:7.1f}% '
          f'[{lo:6.1f}, {hi:6.1f}]{flag}')
