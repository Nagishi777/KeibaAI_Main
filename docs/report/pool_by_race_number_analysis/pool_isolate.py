"""「後半のレースほど厚い」は本当か、R11（メイン）効果と切り分ける。"""
import glob
import os

import pandas as pd

fs = sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv'))
rows = []
for f in fs:
    if os.path.basename(f)[:6] < '202301':
        continue
    d = pd.read_csv(f, encoding='utf-8-sig', low_memory=False,
                    usecols=['race_id', 'date', 'hyosu_total_1m'])
    rows.append(d.groupby('race_id').agg(
        date=('date', 'first'), p1=('hyosu_total_1m', 'first')).reset_index())
g = pd.concat(rows, ignore_index=True)
g['race_id'] = g.race_id.astype(str)
g['raceno'] = g.race_id.str[-2:].astype(int)
g['date'] = pd.to_datetime(g.date)
g['pool_oku'] = g.p1 * 100 / 1e8
g = g[g.pool_oku.notna() & (g.pool_oku > 0)].copy()

print('=== [D-1] R11 を除いた「レース番号 → 総票数」の素の傾向 ===')
sub = g[g.raceno != 11]
t = sub.groupby('raceno')['pool_oku'].agg(n='size', median='median')
t['vs_R1'] = (t['median'] / t.loc[1, 'median']).round(2)
print(t.round(2).to_string())
print()

print('=== [D-2] 1日の中で正規化（R11を除外、日ごとの中央値で割る） ===')
sub = sub.copy()
sub['day_med'] = sub.groupby(sub.date.dt.date)['pool_oku'].transform('median')
sub['rel'] = sub.pool_oku / sub.day_med
t = sub.groupby('raceno')['rel'].agg(n='size', median='median')
print(t.round(3).to_string())
print()

print('=== [D-3] 同じ日・同じ競馬場の中で「R1→R12」が単調に増えるか ===')
# 日×競馬場ごとに Spearman 相関（レース番号 vs 総票数）を取る
g['venue'] = g.race_id.str[4:6]
corrs = []
for _, grp in g[g.raceno != 11].groupby([g.date.dt.date, 'venue']):
    if len(grp) >= 8:
        corrs.append(grp['raceno'].corr(grp['pool_oku'], method='spearman'))
s = pd.Series(corrs).dropna()
print(f'  開催日×競馬場 {len(s)} 件の Spearman 相関（R11除く）')
print(f'    中央値 {s.median():+.3f} / 平均 {s.mean():+.3f}')
print(f'    正の相関（後半ほど厚い）  {(s > 0).mean() * 100:5.1f}%')
print(f'    強い正の相関 (>= +0.5)    {(s >= 0.5).mean() * 100:5.1f}%')
print(f'    負の相関                  {(s < 0).mean() * 100:5.1f}%')
print()

print('=== [D-4] R11含む全レースでの同様の相関 ===')
corrs = []
for _, grp in g.groupby([g.date.dt.date, 'venue']):
    if len(grp) >= 8:
        corrs.append(grp['raceno'].corr(grp['pool_oku'], method='spearman'))
s2 = pd.Series(corrs).dropna()
print(f'  中央値 {s2.median():+.3f} / 正の相関 {(s2 > 0).mean() * 100:.1f}%')
print()

print('=== [D-5] 前半(R1-6) vs 後半(R7-12) ※R11除く ===')
a = g[(g.raceno <= 6)]['pool_oku']
b = g[(g.raceno >= 7) & (g.raceno != 11)]['pool_oku']
print(f'  前半 R1-6    中央値 {a.median():.2f}億 (n={len(a)}) 5億以上 {(a >= 5).mean() * 100:.1f}%')
print(f'  後半 R7-12   中央値 {b.median():.2f}億 (n={len(b)}) 5億以上 {(b >= 5).mean() * 100:.1f}%')
print(f'  R11 のみ     中央値 {g[g.raceno == 11]["pool_oku"].median():.2f}億')
