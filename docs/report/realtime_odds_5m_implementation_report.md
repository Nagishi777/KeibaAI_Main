# 当日オッズ取得の拡張と 5m 判断への移行 実装レポート

作成日: 2026-09-23

## 1. 概要

当日オッズ取得（`src/scraping/realtime_odds.py`）と当日予測（`src/simulator/`）に、次の3点を実装した。

| # | 依頼内容 | 実装結果 |
| --- | --- | --- |
| 1 | 取得時点に `2m` / `3m` / `4m` を追加 | 取得時点を 6 → 9 時点（`60m, 30m, 10m, 5m, 4m, 3m, 2m, 1m, 10s`）に拡張 |
| 2 | 当日予測を 5m オッズで行い、特徴量を `odds_5m`, `inc_share_10m_5m` の2列にする | 特徴量・賭け判断オッズ・レース選別（pool フィルタ）をすべて 5m 時点に変更 |
| 3 | 出力を日付フォルダ＋レース別 CSV（`{race_id}_{tansho\|fukusho\|wakuren}_realtimeodds.csv`）にする | 1レース・1券種につき1ファイルに全時点を追記する形式に変更 |

判断時点を 5m に変更したため、自動購入モジュール（`src/buying/`）が前提にしていた「1m スナップショット」も 5m に合わせた（§5）。

## 2. 取得時点の追加

`DEFAULT_SNAPSHOT_SPECS` に `4m` / `3m` / `2m` を追加した。

```text
60m → 30m → 10m → 5m → 4m → 3m → 2m → 1m → 10s
```

- 同じ時刻に期限を迎えたジョブの処理順（発走に近い時点を優先）は、これまでラベルごとに手書きの辞書で持っていた。これを `DEFAULT_SNAPSHOT_SPECS` から自動生成する `SNAPSHOT_PRIORITY` に変更し、今後時点を追加しても修正が1箇所で済むようにした。
- 鮮度上限（`saved_stale` 判定）は既存ルールのままで、`2m`〜`4m` には発走直前向けの 180 秒が適用される（`60m` / `30m` のみ 900 秒）。
- 先行取得（各基準時刻の2分前から30秒間隔）も既存ロジックのまま全時点に適用される。1分おきに基準時刻が並ぶため、5m〜10s の区間は実質的に連続ポーリングになる。

## 3. 出力形式の変更

### 3.1 新しいフォルダ構成

```text
data/processed/realtime_odds/
└── YYYYMMDD/
    ├── {race_id}_tansho_realtimeodds.csv
    ├── {race_id}_fukusho_realtimeodds.csv
    ├── {race_id}_wakuren_realtimeodds.csv
    └── YYYYMMDD_scheduler_events.csv      ← 取得イベント記録も日付フォルダへ移動
```

- 1ファイルに `60m`〜`10s` の全時点が縦持ち（1行 = 1時点 × 1馬番／組番）で入り、`snapshot_label` 列で時点を区別する。
- 時点ごとに同じファイルへ追記（upsert）し、`target_datetime` → 馬番（組番）の順に並べ直して保存する。重複判定キーは `race_id` + 馬番（組番）+ `snapshot_label` + `target_datetime`。
- ファイル名の `race_id` は出馬表（スケジュール）側の `race_id` を使う。

### 3.2 オッズと票数（votes）について

JV-Link の O1 レコードには**馬ごとの票数は含まれない**。記録している票数は既存と同じく次の列である。

| 列 | 内容 |
| --- | --- |
| `odds_win` / `odds_place_min`・`odds_place_max` / `odds_wakuren` | 各券種のオッズ |
| `hyosu_total` | その券種のレース全体の票数合計（100円単位） |
| `ninkijun` | 人気順 |

馬ごとの投票額が必要な場合は、予測側と同じく `src.simulator.features.estimate_votes`（`0.8 × hyosu_total ÷ odds_win`）で単勝オッズから逆算する。CSV に逆算値の列は追加していない。

### 3.3 追記時の型の修正

既存 CSV を読み直して追記する処理で、馬番 `"01"` が整数 `1` として読まれ、新しい行 `"01"` と重複判定が一致しない問題があった（1ファイルに複数時点を追記する新形式では必ず起きる）。読み直し時に `umaban` / `kumiban` を文字列として読むよう修正した。

## 4. 当日予測の 5m 化

### 4.1 特徴量

`src/simulator/features.py` に起点時点と判断時点を定数として定義し、列名はすべてここから組み立てるようにした。

```python
BASE_SNAPSHOT = '10m'
DECISION_SNAPSHOT = '5m'
FEATURE_COLS = ('odds_5m', 'inc_share_10m_5m')
POOL_COL = 'pool_5m'
```

| 列 | 定義 |
| --- | --- |
| `odds_5m` | 5m 時点の単勝オッズ（生の値） |
| `inc_share_10m_5m` | `inc_i = max(投票額_5m,i − 投票額_10m,i, 0)` をレース内合計で割ったシェア |
| `pool_5m`（特徴量ではなくフィルタ） | 5m 時点の単勝票数合計（円） |

計算式は従来の `inc_share_5m_1m` と同じで、時点だけを 10m→5m にずらしている。

### 4.2 予測フロー（`predict_today.py`）

```text
[1] 10m・5m のオッズ・票数を読込
[2] 馬番別投票額を逆算
[3] 特徴量 odds_5m / inc_share_10m_5m
[4] LightGBM で勝率予測
[5] pool_5m がしきい値未満のレースは見送り
[6] EV = 予測確率 × 5mオッズ ≥ min_ev の馬を買う
[7] fractional Kelly で賭け金決定
```

- 賭け判断オッズ（EV・Kelly 計算）も `odds_5m` に揃えた。特徴量の時点と賭け値の時点を揃えるという spec §6 の方針に従っている。
- レース選別も `pool_5m` に変更した。`pool_1m` のままだと 1m を待つ必要があり、5m に前倒しした意味がなくなるため。
- 出力 CSV（`output/simulator/{YYYYMMDD}_all.csv` / `_bets.csv`）の列は `odds_5m`, `pool_5m`, `inc_share_10m_5m` になる。

### 4.3 当日データの読込（`realtime_loader.py`）

- `data/processed/realtime_odds/YYYYMMDD/*_tansho_realtimeodds.csv` をすべて読み、`snapshot_label` が `10m` / `5m` の行を取り出して、馬ごとに 10m と 5m を横に並べた表（ワイド表）にする。
- 10m・5m のどちらかが欠けた馬は、従来どおり補完せずに除外する（件数はログに出力）。
- `4m` 以降のデータは保存されるが、予測には使わない。

### 4.4 再学習・モデル保存

- `retrain.py` は `REQUIRED_SNAPSHOTS = ('10m', '5m')` で学習データを読み、しきい値は `pool_5m` の分布から求める。
- 運用パラメータ JSON の `odds_snapshot` は `5m` で保存される。

## 5. 自動購入モジュール（`src/buying/`）への影響と対応

購入のリスク判定が「`snapshot_label == "1m"`、基準時刻 = 発走1分前」を固定で要求していたため、5m 判断のままでは全レースが `snapshot_label_mismatch` で見送りになる。次のように変更した。

| ファイル | 変更 |
| --- | --- |
| `snapshot_reader.py` | 判断時点 `DECISION_SNAPSHOT_LABEL` をシミュレータの `DECISION_SNAPSHOT` から取得。イベント CSV は日付フォルダを優先して探す（旧パスも引き続き探す） |
| `risk.py` | 要求ラベルを `5m`、期待基準時刻を「発走5分前」に変更 |
| `strategy.py` | 期待オッズ・メタデータの列を `odds_5m` / `pool_5m` / `inc_share_10m_5m` に変更 |

これは `src/buying/README.md` §2 の選択肢3「1分前データは事後検証専用とし、実購入は別の早いスナップショットで行う」に当たる。既定の判断タイミング（発走3分前）・送信期限（発走2分前）の時点では、5m スナップショットは保存が完了している。

## 6. 利用者への影響（要対応）

1. **再学習が必要。** 既存モデルは `odds_1m` / `inc_share_5m_1m` で学習されている。`load_params` が特徴量の不一致を検出して予測を停止するため、そのまま誤った予測が出ることはないが、`python -m src.simulator.retrain` で作り直す必要がある。
2. **再学習スクリプトは現在のリポジトリだけでは動かない。** `retrain.py` が使う `src.features.market.odds_series_loader` と `src.backtest.dataset` がリポジトリに含まれていない（今回の変更以前から）。学習用の月別オッズデータに `10m` 時点が含まれている必要もある（`docs/report/final_votes_backtest_report.md` を見る限り、60m/30m/10m/5m は存在する）。
3. **旧形式の CSV は読めない。** `data/processed/realtime_odds/YYYYMMDD_tansho_1m.csv` のような旧ファイルは、新しい予測処理からは読まれない。必要であれば変換スクリプトを別途用意する。
4. **5m 条件の成績は未検証。** `pool_filter_condition_spec.md` の結果（EV ≥ 1.10、pool q0.75 など）は 1m 判断での検証値である。5m 判断で同じしきい値が妥当かは、再学習後にバックテストで確認する必要がある。

## 7. テスト

`.venv` に pytest が入っていないため、`unittest` で実行した。

```text
python -m unittest tests.test_realtime_odds tests.test_jvlink_o1 tests.test_realtime_loader tests.test_buying
Ran 24 tests ... OK
```

追加・変更したテスト:

- `test_realtime_odds.py`
  - 出力先が `YYYYMMDD/{race_id}_tansho_realtimeodds.csv`、イベント記録が日付フォルダ内であること
  - 同じレースの 5m / 2m / 1m が1ファイルに追記され、馬番 `"01"` が重複せずに保たれること
  - ジョブが `60m`〜`10s` の9時点で生成され、`3m` の基準時刻が発走3分前になること
- `test_realtime_loader.py`（新規）
  - レース別 CSV から `odds_5m` / `inc_share_10m_5m` / `pool_5m` が期待値どおりに計算されること
  - 5m が欠けたレースが除外されること、日付フォルダが無い場合はエラーになること
- `test_buying.py`
  - 証跡を 5m（発走5分前）に変更。イベント CSV を日付フォルダから読めること、期待オッズに `odds_5m` が入ること

**未確認:** JV-Link 実機での取得（実際の開催日の実行）と、学習済みモデルを使った予測の一連の動作は確認していない。

## 8. 変更ファイル一覧

| ファイル | 内容 |
| --- | --- |
| `src/scraping/realtime_odds.py` | 時点追加、レース別 CSV 出力、日付フォルダ、追記時の型修正 |
| `src/simulator/features.py` | 判断時点の定数化、特徴量を `odds_5m` / `inc_share_10m_5m` に変更 |
| `src/simulator/realtime_loader.py` | レース別 CSV から 10m / 5m を読む形式に書き換え |
| `src/simulator/predict_today.py` | 賭けオッズ・出力列を 5m に変更 |
| `src/simulator/retrain.py` / `artifacts.py` / `__init__.py` | しきい値・`odds_snapshot` を 5m に変更 |
| `src/buying/snapshot_reader.py` / `risk.py` / `strategy.py` / `cli.py` | 5m 判断に合わせた証跡チェック・列名 |
| `tests/test_realtime_odds.py` / `tests/test_buying.py` | 新仕様に更新 |
| `tests/test_realtime_loader.py` | 新規 |
| `docs/manual/scraper_manual.md` / `src/buying/README.md` | 出力パス・時点の記載を更新 |
