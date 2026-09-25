# リアルタイムオッズ取得マニュアル

このマニュアルは、`src/scraping/realtime_odds.py` を使って、JRA-VAN JV-Link の当日オッズをレース発走時刻に合わせて保存する方法を説明します。

## 1. 何ができるか

プログラムを一度起動すると、指定日の全競馬場・全レースの発走時刻を読み込み、各レースについて次の9時点で単勝・複勝・枠連のオッズを取得します。

| ラベル | 発走時刻から |
| --- | ---: |
| `60m` | 60分前 |
| `30m` | 30分前 |
| `10m` | 10分前 |
| `5m` | 5分前 |
| `4m` | 4分前 |
| `3m` | 3分前 |
| `2m` | 2分前 |
| `1m` | 1分前 |
| `10s` | 10秒前 |

起動後はプロセスが終了せず、時刻になるまで待機して順次保存します。端末を閉じたり、`Ctrl+C` を押したりすると停止します。

通常実行では速報オッズ (`0B31`) を各基準時刻の2分前から30秒間隔で先行取得し、基準時刻 (`target_datetime`) 以前にJV-Linkが発表した最新の値を保存します。そのため、基準時刻直前の値を取り込みつつ、取得処理が基準時刻の少し後に動いても未来のオッズが過去時点のCSVへ混入しにくい構成です。JV-Linkの発表時刻は分単位なので、秒単位の厳密な境界は保証されません。

## 2. 前提とインストール

- Windows
- Python 3.11～3.13
- JRA-VAN DataLab と JV-Link（利用キーを設定済み）
- 対象PCからJRAおよびJRA-VANへ接続できること

プロジェクトのルートで仮想環境を作成し、依存パッケージをインストールします。

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirement.txt
python -m playwright install chromium
```

`pywin32` とJV-Link COMはWindows専用です。`JVInit failed` が出る場合は、JV-Linkのインストール、利用キー、起動中の別JV-Linkクライアントを確認してください。

## 3. 実行コマンド

### 通常実行

```powershell
python -m src.scraping.realtime_odds
```

対象日は常に日本時間の当日です。起動時にJRA出馬表を毎回取得して保存し、その内容から全競馬場・全レースのジョブを作成します。過去日や未来日の指定、既存の出馬表CSVの再利用はできません。

### よく使うオプション

| オプション | 内容 |
| --- | --- |
| `--output-dir PATH` | オッズCSVの保存先（既定: `data/processed/realtime_odds`） |
| `--headed` | 出馬表取得用のChromiumを表示（通常は不要） |
| `--sid ID` | `JVInit` に渡すソフトウェアID（既定: `KEIBA_AI`） |
| `--collection-delay-seconds N` | 基準時刻後にJV-Linkを読む待機秒数（既定: 2） |
| `--poll-interval-seconds N` | 基準時刻前の速報オッズ先行取得間隔（既定: 30秒） |
| `--poll-lookback-seconds N` | 基準時刻の何秒前から先行取得するか（既定: 120秒） |
| `--include-past` | 起動時点を過ぎた基準時刻も履歴から復元を試行 |

## 4. いつ起動するか

開催日の朝8:30に起動してください。最初の `60m` ジョブとJRA出馬表取得に必要な時間を確保できます。3競馬場開催の日も、1回の起動で取得した全競馬場・全レースを処理します。

起動が遅れた場合、過ぎたジョブは既定では `skipped_at_startup` として記録されます。過去時点を時系列オッズ (`0B41`) から復元したい場合だけ、次のように `--include-past` を付けます。時系列オッズは5～10分程度の粒度なので、`1m`や`10s`の正確な復元は保証されません。

```powershell
python -m src.scraping.realtime_odds --include-past
```

Windowsタスクスケジューラを使う場合は、プログラムに `.venv\Scripts\python.exe`、引数に `-m src.scraping.realtime_odds`、開始場所にプロジェクトルートを指定し、開催日の朝8:30をトリガーにします。JV-Link COMの利用環境によっては、最初は「ユーザーがログオンしている場合のみ実行」で設定してください。

## 5. 保存されるファイル

### 出馬表（発走時刻）

`data/processed/schedules/YYYYMMDD_jra_today_schedule.csv`

主な列は次のとおりです。

| 列 | 意味 |
| --- | --- |
| `date` | レース日（ISO形式） |
| `time` | 発走時刻（日本時間、`HH:MM`） |
| `post_datetime` | 発走日時 |
| `race_id` | JRA形式のレース識別子（12桁） |
| `rt_key` | JV-Link速報オッズ取得キー（開催日・競馬場・レース番号） |
| `venue_code` | 競馬場コード（通常`01`～`10`） |
| `kai` / `day` | 開催回 / 開催日 |
| `race_number` | レース番号 |
| `race_name` | レース名（取得できない場合は空欄） |
| `access_d_cname` / `shutuba_url` | JRA出馬表取得用の内部情報 |

`venue_code` でグループ化すると、開催場がすべて入っているか確認できます。

### レース別オッズ

`data/processed/realtime_odds/YYYYMMDD/{race_id}_{券種}_realtimeodds.csv`

日付ごとのフォルダの下に、1レース・1券種につき1ファイルを作ります。1ファイルに `60m`～`10s` の全時点が `snapshot_label` 列付きで縦に並び、基準時刻（`target_datetime`）順に追記されます。

例: `20260920/202605040711_tansho_realtimeodds.csv`, `20260920/202605040711_fukusho_realtimeodds.csv`, `20260920/202605040711_wakuren_realtimeodds.csv`。

券種は `tansho`（単勝）、`fukusho`（複勝）、`wakuren`（枠連）です。共通列と券種固有列は次のとおりです。

| 列 | 意味 |
| --- | --- |
| `race_id` | レース識別子 |
| `rt_key` | JV-Link取得キー |
| `data_kubun` | JV-Dataのデータ区分 |
| `made_date` | O1レコード作成日 |
| `keibajo_code` | O1レコード上の競馬場コード |
| `race_num` | O1レコード上のレース番号 |
| `happyo_datetime` | JV-Linkが発表したオッズ日時 |
| `toroku_tosu` / `syusso_tosu` | 登録頭数 / 出走頭数 |
| `hatsubai_flag` | 発売状態フラグ |
| `hyosu_total` | JV-Data O1の票数合計（単位100円） |
| `umaban` | 馬番（単勝・複勝） |
| `kumiban` | 枠番組合せコード（枠連） |
| `odds_win` | 単勝オッズ |
| `odds_place_min` / `odds_place_max` | 複勝オッズの下限 / 上限 |
| `odds_wakuren` | 枠連オッズ |
| `ninkijun` | 人気順 |
| `venue_code` / `race_number` | スケジュール由来の競馬場 / レース番号 |
| `post_datetime` | 発走日時 |
| `snapshot_label` | `60m`～`10s` の保存時点ラベル |
| `target_datetime` | その時点の基準日時（発走日時－オフセット） |
| `acquired_at` | 実際にJV-Linkを読み取った日時 |
| `source_age_seconds` | 基準日時と採用した発表日時の差（秒） |
| `odds_dataspec` | 取得元（通常実行は`0B31`、履歴復元は`0B41`） |

オッズ値はJV-Dataの10倍整数を10で割った小数です（例: 格納値25 → `2.5`）。未発売・取得不能などは空欄になることがあります。CSVはExcelで開きやすいUTF-8 BOM付きです。

### 実行記録

`data/processed/realtime_odds/YYYYMMDD/YYYYMMDD_scheduler_events.csv`

主な列は `rt_key`, `race_id`, `snapshot_label`, `target_datetime`, `post_datetime`, `acquired_at`, `delay_seconds`, `odds_dataspec`, `latest_happyo_datetime`, `source_age_seconds`, `freshness_limit_seconds`, `status`, `rows_saved` です。

`status` の意味:

- `saved`: 1件以上保存
- `saved_stale`: 保存したが、発表値が鮮度上限より古い
- `no_odds_before_target`: 基準時刻以前のオッズが見つからない
- `jvlink_unavailable`: JV-Linkの取得に失敗
- `skipped_at_startup`: 起動時点ですでに基準時刻を過ぎていた

## 6. データの読み方

例えばあるレースの単勝の10分前データを確認します。

```python
import pandas as pd

df = pd.read_csv(
    "data/processed/realtime_odds/20260920/202605040711_tansho_realtimeodds.csv",
    dtype={"race_id": str, "rt_key": str, "umaban": str},
)
m10 = df[df["snapshot_label"] == "10m"]
print(m10[["race_id", "umaban", "odds_win", "ninkijun", "happyo_datetime"]].head())
```

当日予測（`src.simulator.predict_today`）は、この単勝ファイルの `10m` と `5m` の行を使います。

同じレース・同じ馬の時点比較は `race_id` と `umaban`（枠連なら `kumiban`）で行います。発走時刻との比較には `target_datetime`、JV-Link側の発表時刻の確認には `happyo_datetime`、値の古さには `source_age_seconds`、実際の取得遅延の確認には `acquired_at` と `delay_seconds` を使います。

## 7. トラブルシューティング

- **競馬場が1場しかない**: プログラムを再起動して出馬表を取り直し、`*_jra_today_schedule.csv` の `venue_code` を確認します。
- **`skipped_at_startup` が多い**: 最初のレースの70～75分前に起動するか、履歴復元が必要な場合だけ `--include-past` を使います。
- **`no_odds_before_target`**: その時点ではまだJV-Linkに発表値がない、または発売対象外の可能性があります。
- **`saved_stale`**: JV-Linkのキャッシュや配信間隔により、直前3分（60m・30mは15分）以内の値を取得できませんでした。CSVの`happyo_datetime`と`source_age_seconds`を確認してください。
- **Playwrightのエラー**: `python -m playwright install chromium` を再実行します。
- **JV-Link/COMのエラー**: Windows上で実行し、JRA-VAN DataLabの利用キーとJV-Linkの動作を確認します。
- **途中で止めた**: 同じコマンドを再実行すると、既存CSVに同じキーの行を重複させず追記・更新します。

## 8. ファイル構成

- `src/scraping/realtime_odds.py`: 利用者が実行する唯一の公開CLI
- `src/scraping/_jra_today_schedule.py`: JRA出馬表から発走時刻を取得する補助モジュール
- `src/scraping/_jvlink_o1.py`: JV-Link O1レコードを解析する補助モジュール
- `src/scraping/_netkeiba_today_fetcher.py`: 既存の補助取得処理（通常は直接実行しない）

通常は `python -m src.scraping.realtime_odds ...` だけを実行してください。
