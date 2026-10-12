"""総票数とレース番号は相関している。どちらが「本物」かを切り分ける。"""
import glob
import logging
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
logging.basicConfig(level=logging.ERROR)

from src.cli_common import build_config
from src.simulator.features import (
    POOL_COL, TRAIN_DECISION_SNAPSHOT, build_features,
    drop_incomplete_rows, normalize_late_columns, required_snapshots,
)
from src.simulator.odds_series_loader import load_odds_series
from src.simulator.predict_today import predict_proba
from src.simulator.strategy import AnabaRule, apply_rule, REASON_BET

cfg = build_config('config.yaml')
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


def boot(bets, n_boot=4000):
    if len(bets) < 5:
        return (float('nan'),) * 3
    pnl = (bets.won * bets.final_odds * 100).to_numpy()
    pt = pnl.sum() / (len(bets) * 100) * 100
    idx = np.arange(len(bets))
    sims = [pnl[rng.choice(idx, len(idx), replace=True)].sum() / (len(idx) * 100) * 100
            for _ in range(n_boot)]
    return pt, np.percentile(sims, 2.5), np.percentile(sims, 97.5)


print('=== [G-1] 採用点の中での「総票数 vs レース番号」の相関 ===')
print(f'  Spearman = {b.raceno.corr(b.pool_oku, method="spearman"):+.3f}')
print('  （正に相関しているので、片方の効果が他方に見えている恐れがある）')
print()

print('=== [G-2] レース番号を固定して、総票数の効果だけを見る ===')
print('  同じレース番号帯の中で総票数の高低で分ける')
for label, m in [('R1-6 (前半)', b.raceno <= 6), ('R7-12 (後半)', b.raceno >= 7)]:
    sub = b[m]
    med = sub.pool_oku.median()
    lo_s, hi_s = sub[sub.pool_oku < med], sub[sub.pool_oku >= med]
    pl, *_ = boot(lo_s)
    ph, *_ = boot(hi_s)
    print(f'  {label:14s} 中央値{med:5.2f}億 | '
          f'薄い側 {len(lo_s):3d}点 {pl:6.1f}% / 厚い側 {len(hi_s):3d}点 {ph:6.1f}%')
print()

print('=== [G-3] 総票数を固定して、レース番号の効果だけを見る ===')
b['pband'] = pd.cut(b.pool_oku, [0, 5, 7, 10, 1e9],
                    labels=['<5億', '5-7億', '7-10億', '10億+'])
for c in ['<5億', '5-7億', '7-10億', '10億+']:
    sub = b[b.pband == c]
    if len(sub) < 20:
        continue
    e, l = sub[sub.raceno <= 6], sub[sub.raceno >= 7]
    pe, *_ = boot(e)
    pl, *_ = boot(l)
    print(f'  {c:7s} | 前半 {len(e):3d}点 {pe:6.1f}% / 後半 {len(l):3d}点 {pl:6.1f}%')
print()

print('=== [G-4] 結論の確認: 的中率（オッズに依存しない指標）で見る ===')
print('  回収率は少数の高配当で跳ねるので、的中率でも傾向を確認する')
for label, m in [('総票数 <5億', b.pool_oku < 5), ('総票数 >=5億', b.pool_oku >= 5),
                 ('総票数 >=7億', b.pool_oku >= 7),
                 ('R1-6', b.raceno <= 6), ('R7-12', b.raceno >= 7)]:
    s = b[m]
    print(f'  {label:14s} {len(s):4d}点  的中 {s.won.mean() * 100:5.2f}%')
