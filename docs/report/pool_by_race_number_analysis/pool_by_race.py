"""1日の中でレース番号ごとに総票数がどう変わるかを調べる（既存データのみ）。"""
import glob
import os

import pandas as pd

fs = sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv'))
print('files:', len(fs), os.path.basename(fs[0]), '..', os.path.basename(fs[-1]))

rows = []
for f in fs:
    d = pd.read_csv(
        f, encoding='utf-8-sig', low_memory=False,
        usecols=['race_id', 'date', 'hyosu_total_5m', 'hyosu_total_1m'],
    )
    rows.append(
        d.groupby('race_id')
        .agg(date=('date', 'first'),
             p5=('hyosu_total_5m', 'first'),
             p1=('hyosu_total_1m', 'first'),
             n_horses=('hyosu_total_1m', 'size'))
        .reset_index()
    )
g = pd.concat(rows, ignore_index=True)
g['race_id'] = g.race_id.astype(str)
g['venue'] = g.race_id.str[4:6]
g['raceno'] = g.race_id.str[-2:].astype(int)
g['date'] = pd.to_datetime(g.date)
g['year'] = g.date.dt.year
g['pool1_oku'] = g.p1 * 100 / 1e8
g['pool5_oku'] = g.p5 * 100 / 1e8
g = g[g.pool1_oku.notna() & (g.pool1_oku > 0)]

print('races:', len(g), '| date', g.date.min().date(), '->', g.date.max().date())
print()

print('=== [1] 1m総票数（億円）: レース番号別・全期間 ===')
t = g.groupby('raceno')['pool1_oku'].agg(
    n='size', median='median', mean='mean',
    q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75))
print(t.round(2).to_string())
print()

print('=== [2] 2023年以降のみ（1m が信頼できる期間）===')
r = g[g.year >= 2023]
t2 = r.groupby('raceno')['pool1_oku'].agg(
    n='size', median='median', mean='mean',
    q25=lambda s: s.quantile(.25), q75=lambda s: s.quantile(.75),
    pct_over_5oku=lambda s: (s >= 5).mean() * 100)
print(t2.round(2).to_string())
print()

print('=== [3] 同じ日の中での比較（1日ごとに中央値で正規化） ===')
# 日ごとの規模差を除いて「レース番号の効果」だけを見る
r = r.copy()
r['day_median'] = r.groupby(r.date.dt.date)['pool1_oku'].transform('median')
r['rel'] = r.pool1_oku / r.day_median
t3 = r.groupby('raceno')['rel'].agg(n='size', median='median', mean='mean')
print(t3.round(3).to_string())
