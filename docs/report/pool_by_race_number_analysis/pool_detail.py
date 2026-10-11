"""レース番号効果の正体を切り分ける（既存データのみ）。

[A] R11 の跳ね上がりは「時間帯」か「メインレース（重賞）」か
[B] 競馬場・曜日・頭数の影響
[C] 5億円フィルタをレース番号別に見たときの通過率
"""
import glob
import os

import numpy as np
import pandas as pd

fs = sorted(glob.glob('data/processed/odds_series/*_odds_series_tansho.csv'))
rows = []
for f in fs:
    per = os.path.basename(f)[:6]
    if per < '202301':          # 1m が信頼できるのは2023年以降
        continue
    d = pd.read_csv(
        f, encoding='utf-8-sig', low_memory=False,
        usecols=['race_id', 'date', 'post_datetime',
                 'hyosu_total_5m', 'hyosu_total_1m'],
    )
    rows.append(
        d.groupby('race_id')
        .agg(date=('date', 'first'), post=('post_datetime', 'first'),
             p5=('hyosu_total_5m', 'first'), p1=('hyosu_total_1m', 'first'),
             n_horses=('hyosu_total_1m', 'size'))
        .reset_index()
    )
g = pd.concat(rows, ignore_index=True)
g['race_id'] = g.race_id.astype(str)
g['venue'] = g.race_id.str[4:6]
g['raceno'] = g.race_id.str[-2:].astype(int)
g['date'] = pd.to_datetime(g.date)
g['post'] = pd.to_datetime(g.post, errors='coerce')
g['pool1_oku'] = g.p1 * 100 / 1e8
g['pool5_oku'] = g.p5 * 100 / 1e8
g = g[g.pool1_oku.notna() & (g.pool1_oku > 0)].copy()
g['weekday'] = g.date.dt.dayofweek          # 5=土 6=日
g['hour'] = g.post.dt.hour

VENUES = {'01': '札幌', '02': '函館', '03': '福島', '04': '新潟', '05': '東京',
          '06': '中山', '07': '中京', '08': '京都', '09': '阪神', '10': '小倉'}
g['venue_name'] = g.venue.map(VENUES).fillna(g.venue)

print('races:', len(g), '| date', g.date.min().date(), '->', g.date.max().date())
print()

# --- [A] 発走時刻 vs レース番号 ------------------------------------------
print('=== [A-1] 発走時刻(時)別の1m総票数（億円）===')
t = g.groupby('hour')['pool1_oku'].agg(n='size', median='median')
print(t.round(2).to_string())
print()

print('=== [A-2] 同じ発走時刻帯の中で R11 と他レースを比べる ===')
# R11 が「遅い時間だから厚い」のか「メインだから厚い」のかを切り分ける
for h in sorted(g.hour.dropna().unique()):
    sub = g[g.hour == h]
    r11 = sub[sub.raceno == 11]['pool1_oku']
    oth = sub[sub.raceno != 11]['pool1_oku']
    if len(r11) < 20 or len(oth) < 20:
        continue
    print(f'  {int(h)}時台: R11 中央値 {r11.median():6.2f}億 (n={len(r11):4d}) / '
          f'R11以外 {oth.median():6.2f}億 (n={len(oth):4d}) '
          f'-> 倍率 {r11.median() / oth.median():.2f}')
print()

# --- [B] 競馬場・曜日・頭数 ---------------------------------------------
print('=== [B-1] 競馬場別（1m総票数 億円）===')
t = g.groupby('venue_name')['pool1_oku'].agg(
    n='size', median='median',
    pct_over5=lambda s: (s >= 5).mean() * 100).sort_values('median')
print(t.round(2).to_string())
print()

print('=== [B-2] 曜日別 ===')
wd = {5: '土', 6: '日'}
t = g.groupby('weekday')['pool1_oku'].agg(n='size', median='median',
                                          pct_over5=lambda s: (s >= 5).mean() * 100)
t.index = [wd.get(i, f'平日{i}') for i in t.index]
print(t.round(2).to_string())
print()

print('=== [B-3] 出走頭数別 ===')
t = g.groupby('n_horses')['pool1_oku'].agg(n='size', median='median')
print(t[t.n >= 30].round(2).to_string())
print()

# --- [C] フィルタ設計の材料 ---------------------------------------------
print('=== [C-1] 5億円フィルタのレース番号別通過率（1m時点）===')
t = g.groupby('raceno')['pool1_oku'].agg(
    n='size', median='median', pct_over5=lambda s: (s >= 5).mean() * 100)
print(t.round(2).to_string())
print()

print('=== [C-2] 「R9以降だけ買う」といった単純ルールの通過レース数 ===')
total_days = g.date.dt.date.nunique()
print(f'  開催日数: {total_days}日 / 全{len(g)}レース')
for label, mask in [
    ('5億円以上（現行）',            g.pool1_oku >= 5),
    ('R9以降のみ',                   g.raceno >= 9),
    ('R9以降 かつ 5億円以上',        (g.raceno >= 9) & (g.pool1_oku >= 5)),
    ('R10以降のみ',                  g.raceno >= 10),
    ('7億円以上（レポート§5.1）',    g.pool1_oku >= 7),
    ('R9以降 または 7億円以上',      (g.raceno >= 9) | (g.pool1_oku >= 7)),
]:
    n = int(mask.sum())
    print(f'  {label:28s} {n:5d}レース ({n / len(g) * 100:5.1f}%) '
          f'= 1日あたり {n / total_days:5.2f}レース')
print()

print('=== [C-3] 5m時点で見た場合（当日の判断は2m=5mより後だが参考）===')
t = g.groupby('raceno')['pool5_oku'].agg(
    n='size', median='median', pct_over5=lambda s: (s >= 5).mean() * 100)
print(t.round(2).to_string())
print()

print('=== [C-4] 5m -> 1m の伸び率（レース番号別）===')
g['growth'] = g.pool1_oku / g.pool5_oku
t = g.groupby('raceno')['growth'].agg(n='size', median='median')
print(t.round(3).to_string())
