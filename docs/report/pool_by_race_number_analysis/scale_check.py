"""当日(0B31)と履歴(0B41)の差は「単なる定数倍」か、それとも別物か。

もし定数倍なら、しきい値もその倍率で換算できる。
もし分布の形まで違うなら換算は危険で、再計測しかない。
"""
import glob
import os

import numpy as np
import pandas as pd

# 当日データ（速報 0B31）
fs = glob.glob('data/processed/realtime_odds/*/*_tansho_realtimeodds.csv')
d = pd.concat([pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
               for f in fs], ignore_index=True)
d['race_id'] = d.race_id.astype(str)
piv = d.groupby(['race_id', 'snapshot_label'])['hyosu_total'].first().unstack()
piv = piv * 100 / 1e8                      # 億円

# 履歴データ（0B41・2023年以降）
rows = []
for f in sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv')):
    if os.path.basename(f)[:6] < '202301':
        continue
    x = pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                    usecols=['race_id', 'hyosu_total_60m', 'hyosu_total_30m',
                             'hyosu_total_10m', 'hyosu_total_5m', 'hyosu_total_1m'])
    rows.append(x.groupby('race_id').first().reset_index())
h = pd.concat(rows, ignore_index=True)
for s in ['60m', '30m', '10m', '5m', '1m']:
    h['h' + s] = h['hyosu_total_' + s] * 100 / 1e8
h = h[h.h1m > 0]

print('=== [A] 同じ時点どうしの中央値を並べる ===')
print(f'  {"時点":>6s} {"履歴0B41(億)":>14s} {"当日0B31(億)":>14s} {"倍率":>8s}')
for s in ['60m', '30m', '10m', '5m', '1m']:
    hv = h['h' + s].median()
    lv = piv[s].median() if s in piv.columns else np.nan
    print(f'  {s:>6s} {hv:14.2f} {lv:14.2f} {hv / lv:8.2f}')
print()
print('  → 倍率が時点によって揃っていなければ「単なる定数倍」ではない。')
print()

print('=== [B] 時点内での「形」の比較: 1m を1としたときの各時点の比 ===')
print('  （スケールを割り算で消すので、形が同じなら一致するはず）')
print(f'  {"時点":>6s} {"履歴":>8s} {"当日":>8s}')
for s in ['60m', '30m', '10m', '5m']:
    hv = (h['h' + s] / h.h1m).median()
    lv = (piv[s] / piv['1m']).median() if s in piv.columns else np.nan
    print(f'  {s:>6s} {hv:8.3f} {lv:8.3f}')
print()

print('=== [C] レース内のばらつき（変動係数）の比較 ===')
# 分布の形が同じなら、ばらつきの相対的な大きさも近いはず
for name, v in [('履歴 1m', h.h1m), ('当日 2m', piv['2m'].dropna()),
                ('当日 1m', piv['1m'].dropna())]:
    print(f'  {name:9s} n={len(v):6d}  中央値 {v.median():7.2f}  '
          f'q25/q75 = {v.quantile(.25):6.2f}/{v.quantile(.75):6.2f}  '
          f'q75/q25 比 {v.quantile(.75) / v.quantile(.25):5.2f}')
print()
print('  → q75/q25 比が近ければ分布の形（相対的な散らばり）は似ている。')
print()

print('=== [D] 当日データのレース番号別（傾向が履歴と同じか）===')
piv2 = piv.copy()
piv2['raceno'] = piv2.index.str[-2:].astype(int)
t = piv2.groupby('raceno')['2m'].median()
print('  当日2m（1開催日・22レースのみ）:')
print('   ', {int(k): round(v, 2) for k, v in t.items()})
print(f'  レース番号との Spearman 相関: '
      f'{piv2["raceno"].corr(piv2["2m"], method="spearman"):+.3f}')
print('  （履歴では +0.773。サンプルが22レースしかないので参考値）')
