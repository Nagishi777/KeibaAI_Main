"""2m時点の総投票額しきい値をどう決めるべきか、既存データで検討する。

[1] 当日(速報0B31)データの実際の分布 — 何億円に置けば何%通るのか
[2] 学習(履歴0B41)の1m分布と当日2m分布の対応づけ（分位点マッチング）
[3] 履歴の 5m->1m の伸びから 2m 相当を推定し、1m の5億円が 2m では何億円か
"""
import glob
import os

import numpy as np
import pandas as pd

# --- [1] 当日データ（速報 0B31）の 2m 分布 --------------------------------
fs = glob.glob('data/processed/realtime_odds/*/*_tansho_realtimeodds.csv')
d = pd.concat([pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
               for f in fs], ignore_index=True)
d['race_id'] = d.race_id.astype(str)
live = (d[d.snapshot_label == '2m']
        .groupby('race_id')['hyosu_total'].first() * 100 / 1e8)
live = live[live > 0]

print('=== [1] 当日データ（速報0B31）の2m総投票額の分布 ===')
print(f'  レース数: {len(live)}  開催日: '
      f'{sorted(set(os.path.basename(os.path.dirname(f)) for f in fs))}')
print(f'  {"分位点":>8s} {"億円":>8s}')
for q in [0, .1, .25, .5, .75, .9, 1.0]:
    print(f'  {"q%.2f" % q:>8s} {live.quantile(q):8.2f}')
print()
print('  しきい値を置いたときの通過率（当日データ）:')
for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 5.0]:
    print(f'    >= {thr:4.1f}億円 -> {(live >= thr).mean() * 100:5.1f}% '
          f'({int((live >= thr).sum()):3d}/{len(live)}レース)')
print()

# --- [2] 履歴データの 1m / 5m 分布 ----------------------------------------
rows = []
for f in sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv')):
    if os.path.basename(f)[:6] < '202301':
        continue
    x = pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                    usecols=['race_id', 'hyosu_total_5m', 'hyosu_total_1m'])
    rows.append(x.groupby('race_id').agg(p5=('hyosu_total_5m', 'first'),
                                         p1=('hyosu_total_1m', 'first')).reset_index())
h = pd.concat(rows, ignore_index=True)
h['pool5'] = h.p5 * 100 / 1e8
h['pool1'] = h.p1 * 100 / 1e8
h = h[(h.pool1 > 0) & (h.pool5 > 0)]

print('=== [2] 履歴データ（0B41・2023年以降）の分布 ===')
print(f'  レース数: {len(h):,}')
print(f'  {"分位点":>8s} {"5m(億)":>9s} {"1m(億)":>9s}')
for q in [0, .1, .25, .5, .75, .9, 1.0]:
    print(f'  {"q%.2f" % q:>8s} {h.pool5.quantile(q):9.2f} {h.pool1.quantile(q):9.2f}')
print()
print(f'  1m で 5億円は履歴の分位点 q{(h.pool1 < 5).mean():.3f} '
      f'（{(h.pool1 >= 5).mean() * 100:.1f}% が通過）')
print()

# --- [3] 1m の 5億円は 2m では何億円に相当するか -------------------------
# 履歴には 2m が無いので、5m->1m の軌跡から 2m を内挿して推定する。
# プールは締切直前に加速するため線形ではない。ここでは2つの仮定で挟む。
print('=== [3] 「1m で5億円」は 2m では何億円に相当するか（履歴から推定）===')
ratio_5_1 = (h.pool5 / h.pool1).median()
print(f'  履歴: 5m/1m の比の中央値 = {ratio_5_1:.3f}'
      f'（= 5m 時点では1m の約{ratio_5_1 * 100:.0f}%）')
print()
print('  2m は 5m と 1m の間。線形内挿なら (5m→1m の3/4 進んだ点):')
lin = h.pool5 + (h.pool1 - h.pool5) * (3 / 4)
r_lin = (lin / h.pool1).median()
print(f'    2m/1m ≈ {r_lin:.3f} -> 1m の5億円は 2m で約 {5 * r_lin:.2f}億円')
print('  締切直前に加速する分を考えると実際はこれより小さい。')
print(f'  実測の当日データでは 2m/1m = '
      f'{(d[d.snapshot_label == "2m"].groupby("race_id")["hyosu_total"].first() / d[d.snapshot_label == "1m"].groupby("race_id")["hyosu_total"].first()).median():.3f}')
print()

# --- [4] 分位点マッチング: 当日データで「履歴の5億円」に対応する水準 ------
print('=== [4] 分位点マッチングで当日データのしきい値を求める ===')
print('  考え方: 履歴1mで5億円が上位X%なら、当日2m分布の上位X%を取る')
pct_pass = (h.pool1 >= 5).mean()
print(f'  履歴1mで5億円以上 = 上位 {pct_pass * 100:.1f}%')
thr_live = live.quantile(1 - pct_pass)
print(f'  当日2m分布の上位 {pct_pass * 100:.1f}% の境界 = {thr_live:.2f}億円')
print()
print('  ただし当日データは22レース・1開催日のみ。この値は当てにならない。')
print(f'  （当日データの開催: '
      f'{len(set(os.path.basename(os.path.dirname(f)) for f in fs))}日 / {len(live)}レース）')
