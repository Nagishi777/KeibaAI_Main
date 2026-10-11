"""当日データが薄いのは「スケールの違い」か「薄い開催だったから」か。

当日データは 2026-09-26（金）の中京(06)・阪神(09) の22レースのみ。
同条件（同じ競馬場・同じ曜日・ローカル開催）で履歴を絞って比べる。
"""
import glob
import os

import pandas as pd

fs = glob.glob('data/processed/realtime_odds/*/*_tansho_realtimeodds.csv')
d = pd.concat([pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
               for f in fs], ignore_index=True)
d['race_id'] = d.race_id.astype(str)
live = d.groupby(['race_id', 'snapshot_label'])['hyosu_total'].first().unstack() * 100 / 1e8
live['venue'] = live.index.str[4:6]
print('=== 当日データの正体 ===')
print(f'  レース数 {len(live)} / 競馬場 {sorted(set(live.venue))}')
print(f'  日付フォルダ {sorted(set(os.path.basename(os.path.dirname(f)) for f in fs))}')
print('  競馬場別の 2m 中央値:')
for v, g in live.groupby('venue'):
    print(f'    venue {v}: {len(g):2d}レース 中央値 {g["2m"].median():.2f}億円')
print()

# 履歴を同じ競馬場に絞る
rows = []
for f in sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv')):
    if os.path.basename(f)[:6] < '202301':
        continue
    x = pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                    usecols=['race_id', 'date', 'hyosu_total_5m', 'hyosu_total_1m'])
    rows.append(x.groupby('race_id').agg(date=('date', 'first'),
                                         p5=('hyosu_total_5m', 'first'),
                                         p1=('hyosu_total_1m', 'first')).reset_index())
h = pd.concat(rows, ignore_index=True)
h['race_id'] = h.race_id.astype(str)
h['venue'] = h.race_id.str[4:6]
h['date'] = pd.to_datetime(h.date)
h['weekday'] = h.date.dt.dayofweek
h['pool1'] = h.p1 * 100 / 1e8
h['pool5'] = h.p5 * 100 / 1e8
h = h[h.pool1 > 0]

venues = sorted(set(live.venue))
print('=== 履歴を同じ競馬場（中京06・阪神09）に絞った場合 ===')
sub = h[h.venue.isin(venues)]
print(f'  {len(sub):,}レース  1m中央値 {sub.pool1.median():.2f}億円 '
      f'/ 5m中央値 {sub.pool5.median():.2f}億円')
print()
print('  さらに平日（金曜など）に絞る:')
wd = h[h.venue.isin(venues) & (h.weekday < 5)]
if len(wd):
    print(f'    {len(wd):,}レース  1m中央値 {wd.pool1.median():.2f}億円')
else:
    print('    履歴に該当する平日開催なし')
print()

print('=== 結論の確認: 当日 vs 履歴（同じ競馬場）===')
print(f'  履歴 同競馬場 1m 中央値 : {sub.pool1.median():7.2f}億円')
print(f'  当日        1m 中央値 : {live["1m"].median():7.2f}億円')
print(f'  倍率                  : {sub.pool1.median() / live["1m"].median():7.2f}')
print()
print(f'  履歴 同競馬場 5m 中央値 : {sub.pool5.median():7.2f}億円')
print(f'  当日        5m 中央値 : {live["5m"].median():7.2f}億円')
print(f'  倍率                  : {sub.pool5.median() / live["5m"].median():7.2f}')
print()
print('  → 競馬場を揃えても倍率が8倍前後なら、開催の質ではなく')
print('    データ区分（0B31 vs 0B41）の違いである。')
