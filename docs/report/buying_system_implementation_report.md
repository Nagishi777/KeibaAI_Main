# 即PAT自動購入システム 実装・運用ガイド

作成日: 2026-09-22  
対象: `src/buying/`

## 1. 実装結果

`src/buying/README.md` の設計に基づき、次を実装した。

- `src/simulator/predict_today.py` の単勝推奨を購入指示へ変換
- シミュレータ生成済み `*_bets.csv` からの再実行
- 開催表・1mスナップショット取得状態の照合
- 購入単位、レース単位、日単位、実行単位の上限審査
- Playwrightによるログイン、単勝入力、確認画面検証、送信、受付番号取得
- paper / dry-run / live の3モード
- SQLite台帳による二重購入防止
- 送信後の結果不明を `UNKNOWN` とする自動再送禁止
- JSON Lines監査ログと認証値マスク
- アカウント単位の多重起動防止
- `doctor` / `paper` / `dry-run` / `run-once` / `run-day` / `reconcile` CLI

判断ロジックは購入側へ複製していない。`SimulatorStrategy` が
`src.simulator.predict_today.run_predict_today()` を呼び、同モジュールの
`skip_reason == ""` の行だけを購入候補へ変換する。推奨金額は即PATの購入単位に合わせ、
100円未満を見送り、100円単位へ切り下げる。

## 2. 重要な時間制約

即PATのJRA投票締切は原則発走1分前である。一方、現在の1mスナップショットは
発走1分前付近またはその後に保存される。このため、厳密な1m情報を取得してから
発注することはできない。

実装は安全側に、既定の送信期限を発走2分前としている。したがって現在の
`*_1m.csv` をliveで使うと、通常は `submit_deadline_expired` で見送られる。
これは不具合ではなく、締切後送信を避けるためのfail-closed動作である。

live運用には次のいずれかが必要になる。

1. 推奨: シミュレータの判断時点を2～3分前へ変更する。
2. 1mスナップショットを事後検証専用にし、購入には早い時点の値を使う。
3. 1分前以前に既に利用可能だった値だけを使う別の入力契約を定義する。

`BUYING_SUBMIT_DEADLINE_LEAD_SECONDS` は60秒未満に設定できない。

## 3. セットアップ

PowerShellでプロジェクトルートから実行する。

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirement.txt
python -m playwright install chromium
Copy-Item src\buying\env.example src\buying\.env
```

`requirement.txt` には購入側に加え、シミュレータ実行に必要な
PyYAML、NumPy、scikit-learn、LightGBMを追記した。

`src/buying/.env` に本人の即PAT情報を設定する。

```dotenv
JRA_INET_ID=...
JRA_KANYUSHA_NO=...
JRA_PIN=...
JRA_PARS_NO=...

BUYING_MODE=paper
BUYING_KILL_SWITCH=true
BUYING_HEADLESS=false
```

`.env` は `src/buying/.gitignore` の対象である。リポジトリ、クラウドストレージ、
共有フォルダへ置かないこと。

## 4. 事前条件

シミュレータを直接呼ぶ場合、以下が必要になる。

- `config/config.yaml`
- `config/features.json`
- 学習済みモデル
- 対応するキャリブレータ
- `_simulator.json` 運用パラメータ
- 当日の5m・1m単勝CSV

現在のワークスペースには `config/` と学習済みモデルがないため、シミュレータの
直接実行はこの環境では未検証である。別環境で生成した推奨CSVがある場合は
`--bets-csv` で購入側だけを検証できる。

最初に診断する。

```powershell
python -m src.buying.cli --env src\buying\.env doctor
```

`yaml`、`lightgbm`、`sklearn` がNGの場合は依存関係を導入する。
`credentials` と `kill_switch` のNGは、paper運用中なら意図した状態でもよい。

## 5. 推奨する検証順序

### 5.1 単体テスト

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

購入側は実アカウントを使用せず、Fakeブラウザで次を検証している。

- 100円単位への変換
- 時刻・上限・スナップショット鮮度
- 台帳の状態遷移
- 異なるrun IDでも同じ買い目を重複登録しないこと
- paperでブラウザを生成しないこと
- live成功時の `ACCEPTED`
- 結果不明時の `UNKNOWN`

### 5.2 paper

ブラウザを起動せず、入力・戦略・リスク審査・台帳だけを確認する。
検証時だけ `.env` のkill switchを解除する。

```dotenv
BUYING_MODE=paper
BUYING_KILL_SWITCH=false
```

シミュレータを直接呼ぶ場合:

```powershell
python -m src.buying.cli --env src\buying\.env paper --date 2026-09-22
```

生成済み推奨CSVを使う場合:

```powershell
python -m src.buying.cli --env src\buying\.env paper `
  --date 2026-09-22 `
  --bets-csv output\simulator\20260922_bets.csv
```

### 5.3 dry-run

即PATへログインし、購入確認画面まで操作するが、最終送信は行わない。
初回は必ず `BUYING_HEADLESS=false` とする。

```dotenv
BUYING_MODE=dry-run
BUYING_KILL_SWITCH=false
BUYING_HEADLESS=false
```

```powershell
python -m src.buying.cli --env src\buying\.env dry-run --date 2026-09-22
```

dry-runとliveは台帳上の名前空間が別なので、dry-run済みの買い目がliveを妨げない。

### 5.4 live

設定とコマンドの二重解除が必要である。

```dotenv
BUYING_MODE=live
BUYING_KILL_SWITCH=false
BUYING_HEADLESS=false
```

```powershell
python -m src.buying.cli --env src\buying\.env run-once --live
```

`--live` がなければ購入しない。初回は1点100円、1レース100円、1日100円の上限を維持し、
利用者が画面を確認しながら実行すること。

## 6. 当日連続運転

JV-Linkの取得処理を別プロセスで先に起動する。

```powershell
python -m src.simulator.scraping.realtime_odds
```

購入側を別のPowerShellで起動する。

```powershell
python -m src.buying.cli --env src\buying\.env run-day --live
```

`run-day` は `BUYING_POLL_INTERVAL_SECONDS` ごとにシミュレータと購入処理を呼ぶ。
同一買い目は台帳で事前に除外されるため、再ログイン・再購入しない。
最終レース発走5分後に終了する。

シミュレータの `--fetch` は全スケジュール完了まで待機する実装のため、
当日連続購入では購入CLIの `--fetch` ではなく、上記の別プロセス構成を使う。

## 7. 出力

```text
data/processed/buying/
  buying_ledger.sqlite3
  <account_alias>.lock
  logs/
    YYYYMMDD_buying.jsonl
```

台帳の主な状態:

| 状態 | 意味 |
|---|---|
| `PLANNED` | 台帳へ予約済み |
| `VALIDATED` | リスク審査・確認前検証済み |
| `SUBMITTING` | 最終送信直前に永続化済み |
| `ACCEPTED` | 受付番号・金額を確認済み |
| `UNKNOWN` | 送信後に成立確認不能。自動再送禁止 |
| `REJECTED` | 送信前の画面不一致等 |
| `DRY_RUN` | paperまたはdry-run完了 |

成立不明を確認する。

```powershell
python -m src.buying.cli --env src\buying\.env reconcile
```

現在の `reconcile` は対象一覧の表示までである。即PATの投票内容照会に対する自動照合は、
実アカウントで画面契約を検証できないため実装対象から外した。`UNKNOWN` がある場合は
自動再送せず、投票内容照会を人が確認する。

## 8. 未検証・Skipした範囲

依頼どおり、実購入が必要な次の検証は実施していない。

- 即PATの実アカウントログイン
- 現行画面での馬番・金額入力ロケータ
- 最終「投票」「はい」のクリック
- 受付番号・受付金額の実画面解析
- 投票内容照会との自動照合
- CAPTCHA、追加認証、メンテナンス画面（回避処理は実装しない）

Playwright実装は、要素が一意でない、確認画面のレース・券種・馬番・総額が一致しない、
受付番号を確認できない場合に停止する。画面変更を推測で通過して誤購入しない方針である。

## 9. 実装ファイル

| パス | 役割 |
|---|---|
| `src/buying/config.py` | `.env`読込・設定検証 |
| `src/buying/strategy.py` | simulator接続・金額正規化 |
| `src/buying/snapshot_reader.py` | 開催表・取得状態読込 |
| `src/buying/risk.py` | 上限・締切・鮮度審査 |
| `src/buying/ledger.py` | SQLite・冪等性・状態遷移 |
| `src/buying/service.py` | 購入ユースケース |
| `src/buying/browser/playwright_client.py` | 即PAT画面操作 |
| `src/buying/cli.py` | コマンドライン |
| `tests/test_buying.py` | 非実購入テスト |

