# 即PAT 馬券自動購入システム 設計書

> 実装済み。セットアップ、実行方法、未検証範囲は
> `docs/report/buying_system_implementation_report.md` を参照すること。

## 1. 目的と対象範囲

JRA-VAN/JV-Link から取得済みの発走前オッズ・票数情報を入力にし、外部の判断ロジックが返した購入指示を、Python + Playwright で即PATへ入力・送信・照合する。

本設計の対象は次の範囲とする。

- 当日のJRAレースを対象とした即PATへのログイン
- 購入判断モジュールの呼び出しと、購入指示の検証
- 通常投票のブラウザ入力、最終確認、送信
- 投票結果または投票内容照会による成立確認
- 購入上限、締切、重複購入を含む安全制御
- 監査ログ、障害時の停止・復旧
- dry-run（購入しない検証）と live（実購入）の明確な分離

次は対象外とする。

- 購入対象を選ぶ予測・期待値・資金配分ロジックの中身
- 即PATの新規会員登録
- 銀行口座から即PATへの自動入金および出金
- CAPTCHA、追加認証、メンテナンス画面等の回避
- 地方競馬、海外競馬、WIN5

## 2. 最重要の時間制約

2026年のJRA即PATは、JRAレースの発売締切が原則「発走時刻1分前」である。そのため、**発走1分前になってから得られる情報で判断し、その後に投票する要件は成立しない**。

既存の `src/scraping/realtime_odds.py` が保存する `snapshot_label=1m` は、`target_datetime = post_datetime - 1 minute` 以前にJRA-VANが発表した最新値を、基準時刻付近またはその後に保存する仕組みである。保存完了を待ってから即PATを操作すると締切後になる可能性が高い。

live運用までに、次のいずれかを選ぶ必要がある。

1. 推奨: 判断・投票開始を発走2～3分前に変更し、その時点で取得済みの最新値を使う。
2. 「1分前」をデータの名称として残しつつ、`decision_deadline` までに利用可能な値だけを使い、間に合わなければ必ず見送る。この場合、厳密な1分前最新値ではない。
3. 1分前データは事後検証専用とし、実購入は別の早いスナップショットで行う。

初期値は安全側に以下を推奨する。

```text
decision_at       = 発走3分前
submit_deadline   = 発走2分前
official_close    = 発走1分前（参考値。システムの送信期限には使わない）
```

スケジュール変更、発走時刻変更、通信遅延があるため、画面上の締切表示も確認する。ただし画面の締切表示が取得できない、またはローカル時刻との差が許容値を超えた場合は購入しない。

## 3. 機能要件

### FR-01 設定と秘密情報

- 認証情報はソースコード、CSV、ログへ記録しない。
- `src/buying/.env` をローカルに用意し、`env.example` をひな型とする。
- 必須認証情報は `JRA_INET_ID`、`JRA_KANYUSHA_NO`、`JRA_PIN`、`JRA_PARS_NO` とする。
- `.env` の未設定、形式不正、プレースホルダー値を検出した場合は起動を拒否する。
- JRAは会員情報を外部サイトへ保存しないよう注意喚起しているため、`.env` は利用者管理PC内だけに置き、クラウド同期・リポジトリ登録を禁止する。
- ログ、例外、スクリーンショット、Playwright traceでは入力値をマスクする。

### FR-02 入力データ

既存の以下を利用する。

- レース情報: `data/processed/schedules/YYYYMMDD_jra_today_schedule.csv`
- オッズ・票数: `data/processed/realtime_odds/YYYYMMDD/{race_id}_{bet_type}_realtimeodds.csv`（全時点を `snapshot_label` 列で保持）
- 取得状態: `data/processed/realtime_odds/YYYYMMDD/YYYYMMDD_scheduler_events.csv`

> 2026-09-23 更新: 上記 §2 の選択肢3を採用し、シミュレータの判断時点を `5m` に前倒しした。購入時の証跡チェックは `snapshot_label=5m`・`target_datetime = post_datetime - 5 minutes` を要求する（`src/buying/snapshot_reader.py` の `DECISION_SNAPSHOT_LABEL`）。

判断モジュールへ渡す `DecisionContext` は最低限、次を持つ。

```text
race_id, rt_key, venue_code, race_number, post_datetime
snapshot_label, target_datetime, happyo_datetime, acquired_at
source_age_seconds, odds_dataspec, odds rows
account_balance, already_purchased_amount, decision_at
```

入力は以下を満たさなければならない。

- `race_id` と `rt_key` が対象レースに一致する。
- `snapshot_label` が設定値と一致する。
- オッズ行の `happyo_datetime <= target_datetime`。
- `scheduler_events.status` が許可状態（初期値は `saved` のみ）。
- 欠損、重複、発売停止、取消馬を含む不整合がない。
- 判断時点が `submit_deadline` を過ぎていない。

### FR-03 判断ロジックの境界

判断ロジックは差し替え可能なインターフェースとする。

```python
class BettingStrategy(Protocol):
    def decide(self, context: DecisionContext) -> list[BetIntent]: ...
```

`BetIntent` は次の値だけを返し、ブラウザ操作を直接行わない。

```text
race_id
bet_type       # win/place/quinella/...（初期リリース対象は別途限定）
selection      # 馬番・枠番の正規化済みタプル
amount_yen     # 100円単位
strategy_id
reason_code
```

### FR-04 リスク審査

判断結果は即PATへ渡す前に、決定論的な `RiskGate` で全件検査する。

- 1点100円単位、最小・最大購入額
- 1レース、1日、1節の購入上限
- 1レースの購入点数上限
- 券種の許可リスト
- 残高不足
- 対象レース・開催日の一致
- 出走取消・発売停止
- staleなオッズ・票数
- 締切までの残り時間
- 同じ購入キーの重複
- 総購入停止スイッチ

一部だけを購入すると戦略の意図が変わり得るため、初期仕様は「1レース分を全件承認または全件棄却」とする。

### FR-05 ブラウザ操作

- PlaywrightのChromiumを使用する。
- ログイン、場選択、レース選択、式別選択、馬・枠番選択、金額入力、確認、送信をPage Objectに分離する。
- DOMロケータは表示文言、label、role等を優先し、画面構造依存のCSS/XPathを局所化する。
- 実送信直前に、画面から読み取ったレース、券種、組合せ、金額、総額を購入指示と完全一致検証する。
- `BUYING_MODE=dry-run` では、送信確認の直前まで検証して離脱し、送信ボタンを押さない。
- `BUYING_MODE=live` でのみ最終送信を許可する。

### FR-06 成立確認と二重購入防止

成立は「送信ボタンを押したこと」ではなく、次のいずれかで確認する。

- 投票結果画面の受付番号、受付時刻、受付金額、内容が一致する。
- 結果画面が不明な場合、投票内容照会で同一内容の受付を確認する。

購入の一意キー（idempotency key）は次とする。

```text
sha256(account_alias, race_id, bet_type, normalized_selection,
       amount_yen, strategy_id, decision_run_id)
```

- 送信前にローカル台帳へ `SUBMITTING` を永続化する。
- 受付確認後に `ACCEPTED` と受付番号を保存する。
- 送信後の通信断は `UNKNOWN` とし、**自動再送しない**。投票内容照会で解決する。
- 同じ一意キーが `SUBMITTING`、`UNKNOWN`、`ACCEPTED` の場合は再送しない。

### FR-07 状態管理

購入単位の状態遷移を以下とする。

```text
PLANNED -> VALIDATED -> SUBMITTING -> ACCEPTED
    |          |             |          |
    +------> REJECTED        +-> UNKNOWN
                           
PLANNED/VALIDATED -> EXPIRED
UNKNOWN -> ACCEPTED | NOT_FOUND | MANUAL_REVIEW
```

`UNKNOWN` から `NOT_FOUND` へ移すには、投票内容照会を複数回確認し、照会可能な時間窓と受付内容を検証する。live初期版では自動再送せず `MANUAL_REVIEW` で停止する。

### FR-08 ログと監査

- SQLite台帳を `data/processed/buying/buying_ledger.sqlite3` に保存する。
- JSON Linesの運用ログを `data/processed/buying/logs/` に保存する。
- 記録項目: run ID、判断入力のハッシュ、購入指示、審査結果、状態遷移、締切、受付番号、受付時刻、金額、エラー分類。
- PIN、P-ARS番号、INET-ID、加入者番号、Cookie、セッション、口座情報は記録しない。
- スクリーンショットは原則エラー画面のみとし、認証画面では無効化する。保存時も認証情報が写っていないことを保証できない場合は保存しない。

### FR-09 運用制御

- CLIで `doctor`、`dry-run`、`run-once`、`reconcile` を提供する。
- live実行は設定値に加え、明示引数 `--live` の二重条件を要求する。
- `BUYING_KILL_SWITCH=true` ならログインを含む購入処理を開始しない。
- 連続エラー、DOM不一致、時刻ずれ、残高不一致、照合不能でサーキットブレーカーを開き、その日の自動購入を停止する。
- 同一アカウントで複数プロセスが動かないようOSファイルロックを取る。

## 4. 非機能要件

- 安全性: fail closed。判断不能・画面不一致・通信断では購入しない。
- 再現性: 判断に使った入力ファイルと行をハッシュで追跡できる。
- 可観測性: 各レースの見送り理由と状態遷移を後から確認できる。
- 保守性: JRA画面変更の影響をPage Objectと契約テストへ閉じ込める。
- 時刻: 内部はtimezone-awareなJSTで扱い、naive datetimeを境界で拒否する。
- 性能: 判断開始から送信完了までの時間予算を計測し、締切超過前に中止する。
- 可用性: 自動リトライは送信前の読み取り操作に限定する。送信操作は再実行しない。
- テスト性: 実アカウント不要のHTML fixture、fake strategy、fake clockで大部分を検証できる。

## 5. システム構成

```text
src.scraping.realtime_odds / schedule CSV
                    |
                    v
             SnapshotReader
                    |
                    v
             BettingStrategy  <--- 後日実装する判断ロジック
                    |
                 BetIntent[]
                    |
                    v
                RiskGate  <--- 上限、締切、重複、残高
                    |
              PurchasePlan
                    |
                    v
             PurchaseService
              /     |      \
             v      v       v
       IpatBrowser  Ledger  AuditLog
             |
             v
       即PAT (Playwright)
             |
             v
      Result/Reconciliation
```

実装ファイル構成:

```text
src/buying/
  README.md                 # 本設計書
  env.example               # 秘密値なしの設定ひな型
  __init__.py
  cli.py                    # doctor/dry-run/run-once/reconcile
  config.py                 # env読込・検証
  domain.py                 # BetIntent, PurchasePlan, PurchaseResult
  clock.py                  # JST時刻と締切計算
  snapshot_reader.py        # 既存CSV入力
  strategy.py               # Protocolと仮のNoBetStrategy
  risk.py                   # 購入上限・整合性検証
  ledger.py                 # SQLite、idempotency
  service.py                # ユースケース制御
  browser/
    session.py              # Playwright lifecycle
    login_page.py
    bet_page.py
    confirmation_page.py
    result_page.py
    inquiry_page.py
    errors.py
  audit.py                  # マスク済み監査ログ
  process_lock.py           # アカウント単位の多重起動防止
tests/buying/
  fixtures/                 # 秘密情報を含まないHTML
  test_config.py
  test_clock.py
  test_snapshot_reader.py
  test_risk.py
  test_ledger.py
  test_browser_pages.py
  test_service.py
```

## 6. 実行シーケンス

1. 単一プロセスロックを取得する。
2. 設定、時刻同期、kill switch、発売時間、入力ファイルを検査する。
3. 即PATへログインし、残高と画面上の締切を取得する。
4. 対象レースのスナップショットを読み、鮮度と時刻を検査する。
5. `BettingStrategy.decide()` を呼び、購入指示を得る。
6. `RiskGate` がレース単位で全件審査する。
7. dry-runなら最終送信直前の画面一致を検査して終了する。
8. liveなら購入台帳を `SUBMITTING` にしてから1回だけ送信する。
9. 結果画面を解析し、受付番号と購入内容を照合する。
10. 結果不明なら再送せず、投票内容照会で照合する。
11. 台帳を確定し、レース単位の結果を監査ログへ残す。

## 7. エラー方針

| エラー | 処理 |
| --- | --- |
| 認証失敗・追加認証・CAPTCHA | 即時停止。回避しない |
| DOM/文言変更 | 即時停止。Page Objectを更新するまでlive禁止 |
| オッズ未取得・古い・時刻矛盾 | 当該レースを見送り |
| 締切までの余裕不足 | 当該レースを見送り |
| 残高不足・上限超過 | レース全体を棄却 |
| 送信前の通信失敗 | 時間内ならページを再取得して最初から検証 |
| 送信後の通信失敗 | `UNKNOWN`。自動再送禁止、照会へ |
| 受付内容不一致 | `MANUAL_REVIEW`、以後のlive購入を停止 |
| 連続失敗 | サーキットブレーカーを開き当日停止 |

## 8. テスト・導入段階

1. 単体テスト: 時刻境界、金額、正規化、上限、重複、状態遷移。
2. fixtureテスト: 保存HTMLに対する画面解析。認証情報は使わない。
3. headed dry-run: 実画面で送信直前までを目視確認。
4. paper mode: 実画面へログインせず、入力から購入予定・監査ログだけ生成。
5. 最小額live: 対象レース・券種・1日上限を極小に固定し、利用者立会いで検証。
6. 通常live: 成立照合、停止条件、当日レポートを確認後に限定運用。

自動テストや通常の開発コマンドがlive送信へ到達しないよう、live送信コードは環境変数とCLIフラグの両方がない限り実行不能にする。

## 9. 受入条件

- `.env` がなくてもdry-run/paper用のテストは実行できる。
- 秘密値がログ、例外、trace、SQLiteへ残らない。
- 同一購入指示を2回与えても2回目は送信されない。
- 送信直後にブラウザを切断しても自動再送されない。
- 締切、オッズ鮮度、残高、上限のいずれかが不明なら購入されない。
- dry-runでは即PATへの購入送信が絶対に発生しない。
- liveでは結果画面または内容照会と台帳の金額・内容が一致する。
- JRA画面変更で既知のロケータが取れない場合、誤操作せず停止する。

## 10. live運用前に確定が必要な事項

- 「1分前情報」の厳密な定義と、実投票に使用する判断時刻
- 初期対応する券種（既存取得済みは単勝・複勝・枠連）
- 1点、1レース、1日、1節あたりの上限
- 自動入金を対象外としたままでよいか（推奨は対象外）
- 取消・発走時刻変更の一次情報をどこから取得するか
- live時にheadedブラウザを必須にするか

## 11. 公式仕様上の注意

- 即PATのJRA当日発売は原則として発走1分前に締め切られる。
- JRAは締切直前の混雑や通信障害により申込み・成立確認ができない可能性を案内している。
- 契約成立後の取消・変更はできない。
- 受付番号を確認できなくても成立している場合があり、投票内容照会での確認が必要である。
- JRAは他サイト・アプリケーションとの連携利用時の投票成否・内容を保証していない。画面自動操作は変更・停止リスクを前提にする。
- 利用開始前およびJRA側の仕様変更時には、利用者自身が最新の規約・利用ガイドを確認する。

参考（2026-09-21確認）:

- https://www.jra.go.jp/dento/soku/instructions/hatsubai/jra.html
- https://www.jra.go.jp/dento/soku/instructions.html
- https://www.jra.go.jp/dento/soku/instructions/guide.html
