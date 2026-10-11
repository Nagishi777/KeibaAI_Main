"""Playwrightによる即PAT画面操作。

画面変更を誤操作で吸収しないため、各段階で一意な要素と確認画面の総額を検証する。
ロケータが一致しない場合は推測して送信せず ``BrowserContractError`` で停止する。

このアダプタのlive経路は実アカウントを用いた自動テストを行っていない。
初回はheaded + dry-runで送信確認直前までを利用者が検証すること。
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from typing import Any, Iterable

from src.buying.clock import JST
from src.buying.domain import (
    BetIntent,
    Credentials,
    PreparedPurchase,
    PurchaseReceipt,
    RaceInfo,
)
from src.buying.browser.base import BrowserContractError
from src.buying.browser.diagnostics import ScreenDumper

VENUE_NAMES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}


class PlaywrightIpatClient:
    def __init__(
        self,
        *,
        login_url: str,
        headless: bool,
        timeout_ms: int = 15_000,
        dumper: ScreenDumper | None = None,
        dump_stages: bool = False,
    ) -> None:
        self.login_url = login_url
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.dumper = dumper if dumper is not None else ScreenDumper()
        self.dump_stages = dump_stages
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._prepared_token: str | None = None
        self._submitted = False

    def _ensure_started(self) -> None:
        if self._page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise BrowserContractError("Playwrightがインストールされていません") from exc
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(locale="ja-JP", timezone_id="Asia/Tokyo")
        self._page = self._context.new_page()
        self._page.set_default_timeout(self.timeout_ms)

    def _scopes(self) -> Iterable[Any]:
        yield self._page
        for frame in self._page.frames:
            if frame != self._page.main_frame:
                yield frame

    def _unique(self, locator: Any, description: str) -> Any:
        count = locator.count()
        if count != 1:
            raise self._contract_error(
                f"{description} が一意ではありません（{count}件）", description
            )
        return locator

    def _contract_error(self, message: str, label: str) -> BrowserContractError:
        """画面ダンプを残した上で契約違反エラーを作る。"""
        saved = None
        if self._page is not None:
            frames = [frame for frame in self._scopes() if frame is not self._page]
            saved = self.dumper.dump(self._page, label, scopes=frames)
        if saved is not None:
            return BrowserContractError(f"{message}（画面ダンプ: {saved}）")
        return BrowserContractError(message)

    def _find_input(self, labels: tuple[str, ...], selectors: tuple[str, ...]) -> Any:
        for scope in self._scopes():
            for label in labels:
                locator = scope.get_by_label(re.compile(re.escape(label), re.I))
                if locator.count() == 1:
                    return locator
            for selector in selectors:
                locator = scope.locator(selector)
                if locator.count() == 1:
                    return locator
        raise self._contract_error(
            f"入力欄が見つかりません: {labels}", f"input_{labels[0]}"
        )

    def _click_candidates(self, scope: Any, name: str, exact: bool) -> Iterable[Any]:
        """表示名が name である操作要素の探索候補を、確実な順に返す。

        即PATの送信ボタンはテキストを持たない空の ``<a>`` で、表示名は title
        属性にしかない。role 検索では title もアクセシブル名として扱われるが、
        それだけに頼らず属性指定も試す。
        """
        escaped = name.replace('"', '\\"')
        yield scope.get_by_role("button", name=name, exact=exact)
        yield scope.get_by_role("link", name=name, exact=exact)
        # 即PATは同一のボタンをPC版/スマホ版で二重に持ち、片方をCSSで隠す。
        # 属性セレクタは非表示要素も数えるため :visible で絞る。
        yield scope.locator(f'a[title="{escaped}"]:visible')
        yield scope.locator(f'[aria-label="{escaped}"]:visible')
        yield scope.locator(f'input[type="submit"][value="{escaped}"]:visible')
        yield scope.locator(f'input[type="button"][value="{escaped}"]:visible')
        # 「通常<small>投票</small>」のように語が要素で分割されると、表示名に
        # 空白が入る（"通常 投票"）。語間の空白を無視して突き合わせる。
        spaced = r"\s*".join(re.escape(char) for char in name)
        pattern = re.compile(rf"^\s*{spaced}\s*$")
        yield scope.get_by_role("button", name=pattern)
        yield scope.get_by_role("link", name=pattern)
        yield scope.get_by_text(name, exact=exact)

    def _click_text(self, names: tuple[str, ...], *, exact: bool = True) -> None:
        for scope in self._scopes():
            for name in names:
                for locator in self._click_candidates(scope, name, exact):
                    try:
                        found = locator.count()
                    except Exception:
                        continue
                    if found == 1:
                        locator.click()
                        return
        raise self._contract_error(
            f"操作要素が見つからないか一意ではありません: {names}", f"click_{names[0]}"
        )

    def _body_text(self) -> str:
        texts: list[str] = []
        for scope in self._scopes():
            try:
                texts.append(scope.locator("body").inner_text())
            except Exception:
                continue
        return "\n".join(texts)

    def dump_screen(self, label: str) -> Any:
        """現在の画面を診断用に保存する（セレクタ調査用）。"""
        if self._page is None:
            return None
        frames = [frame for frame in self._scopes() if frame is not self._page]
        return self.dumper.dump(self._page, label, scopes=frames)

    def _dump_stage(self, label: str) -> None:
        if self.dump_stages:
            self.dump_screen(label)

    def login(self, credentials: Credentials) -> None:
        self.dumper.register_secrets(
            credentials.inet_id,
            credentials.subscriber_number,
            credentials.pin,
            credentials.pars_number,
        )
        self._ensure_started()
        self._page.goto(self.login_url, wait_until="domcontentloaded")
        self._dump_stage("stage1_initial")
        inet = self._find_input(
            ("INET-ID", "INET ID"),
            ('input[name*="inet" i]', 'input[id*="inet" i]'),
        )
        inet.fill(credentials.inet_id)
        self._click_text(("ログイン", "送信"))
        self._page.wait_for_load_state("domcontentloaded")
        self._dump_stage("stage2_after_inet")

        subscriber = self._find_input(
            ("加入者番号",),
            ('input[name*="kanyu" i]', 'input[id*="kanyu" i]', 'input[name="i"]'),
        )
        pin = self._find_input(
            ("暗証番号",),
            ('input[name*="pin" i]', 'input[id*="pin" i]', 'input[name="p"]'),
        )
        pars = self._find_input(
            ("P-ARS番号", "P-ARS"),
            ('input[name*="pars" i]', 'input[id*="pars" i]', 'input[name="r"]'),
        )
        subscriber.fill(credentials.subscriber_number)
        pin.fill(credentials.pin)
        pars.fill(credentials.pars_number)
        # 2段目の送信ボタンはテキストのない空の <a> で、表示名は
        # title="ネット投票メニューへ" のみ（「ログイン」では一致しない）。
        self._click_text(("ネット投票メニューへ", "ログイン", "送信"))
        self._page.wait_for_load_state("domcontentloaded")
        self._wait_for_menu()
        self._dump_stage("stage3_after_login")
        text = self._body_text()
        if not any(word in text for word in ("投票", "購入限度額", "メニュー")):
            raise self._contract_error(
                "ログイン後のメニューを確認できません", "login_menu"
            )

    def _wait_for_menu(self) -> None:
        """メニューのAngularJS描画と残高取得の完了を待つ。

        ログイン直後は「NOW LOADING」で、金額セルは単位だけの空表示になる。
        DOM構築は先に終わるので、読み込み表示の消滅ではなく金額の出現を待つ。
        """
        try:
            self._page.wait_for_function(
                "() => /[0-9][0-9,]*\\s*円/.test(document.body.innerText)",
                timeout=self.timeout_ms,
            )
        except Exception:
            # ここで止めず、後続の検証（メニュー語句・残高取得）に判断を委ねる。
            pass

    def get_balance_yen(self) -> int | None:
        """購入限度額を読む。画面に金額欄がなければ ``None``。

        即PATのメニューは非開催日や入金前に金額行そのものを描画しない。
        その場合は残高による上限審査を行わず、他のリスク審査（1日上限・
        レース上限・台帳）に委ねる。
        """
        text = self._body_text()
        # メニューは表組みで、ラベルと金額が別セルに分かれる（間はタブか改行）。
        match = re.search(
            r"(?:購入限度額|残高|ご利用可能額)\s*[:：]?\s*([0-9][0-9,]*)\s*円", text
        )
        if not match:
            return None
        return int(match.group(1).replace(",", ""))

    def prepare_win_bets(
        self, race: RaceInfo, intents: tuple[BetIntent, ...]
    ) -> PreparedPurchase:
        if not intents:
            raise BrowserContractError("空の購入指示は入力できません")
        venue = VENUE_NAMES.get(race.venue_code)
        if venue is None:
            raise BrowserContractError(f"未対応の競馬場コードです: {race.venue_code}")
        self._dump_stage("bet1_menu")
        self._click_text(("通常投票",))
        self._dump_stage("bet2_venue_list")
        self._click_text((venue,))
        self._dump_stage("bet3_race_list")
        self._click_text((f"{race.race_number}R", str(race.race_number)))
        self._dump_stage("bet4_bet_type")
        self._click_text(("単勝",))
        self._dump_stage("bet5_horse_select")

        for intent in intents:
            horse = intent.selection[0]
            self._click_text((str(horse), f"{horse}番"))
        self._dump_stage("bet6_horses_selected")
        self._click_text(("金額入力", "次へ"))
        self._dump_stage("bet7_amount_form")

        for intent in intents:
            self._fill_amount_for_horse(intent.selection[0], intent.amount_yen)
        self._dump_stage("bet8_amounts_filled")
        self._click_text(("セット",))
        self._dump_stage("bet9_after_set")
        self._click_text(("入力終了",))
        self._dump_stage("bet10_confirmation")
        total = sum(item.amount_yen for item in intents)
        self._fill_confirmation_total(total)
        self._dump_stage("bet11_total_filled")
        text = self._body_text()
        try:
            self._validate_summary(text, race, intents, total)
        except BrowserContractError as exc:
            raise self._contract_error(str(exc), "bet12_summary_mismatch") from exc
        token = uuid.uuid4().hex
        self._prepared_token = token
        self._submitted = False
        return PreparedPurchase(
            token=token,
            race_id=race.race_id,
            intents=intents,
            displayed_total_yen=total,
            displayed_summary=text[-4000:],
        )

    def _fill_amount_for_horse(self, horse: int, amount_yen: int) -> None:
        for scope in self._scopes():
            marker = scope.get_by_text(re.compile(rf"^(?:馬番\s*)?0?{horse}$"))
            for index in range(marker.count()):
                row = marker.nth(index).locator("xpath=ancestor::*[self::tr or self::li or @role='row'][1]")
                inputs = row.locator("input:not([type=hidden]):not([type=checkbox]):not([type=radio])")
                if inputs.count() == 1:
                    # 即PATの金額欄は末尾に「00円」が固定表示される形式を想定。
                    inputs.fill(str(amount_yen // 100))
                    return
        raise self._contract_error(
            f"馬番{horse}の金額入力欄を一意に取得できません", f"amount_horse{horse}"
        )

    def _fill_confirmation_total(self, total_yen: int) -> None:
        field = self._find_input(
            ("確認入力", "合計金額"),
            (
                'input[name*="confirm" i]',
                'input[id*="confirm" i]',
                'input[name*="total" i]',
            ),
        )
        field.fill(str(total_yen))

    @staticmethod
    def _validate_summary(
        text: str,
        race: RaceInfo,
        intents: tuple[BetIntent, ...],
        total_yen: int,
    ) -> None:
        compact = re.sub(r"\s+", "", text)
        required = ["単勝", str(total_yen), f"{race.race_number}R"]
        required.extend(str(intent.selection[0]) for intent in intents)
        missing = [value for value in required if value not in compact]
        if missing:
            raise BrowserContractError(f"確認画面の購入内容が一致しません: missing={missing}")

    def cancel_prepared(self, prepared: PreparedPurchase) -> None:
        self._assert_prepared(prepared)
        # 戻る操作による再送を避け、トップ/メニューへ明示遷移する。
        self._click_text(("メニュー", "トップ"))
        self._dump_stage("bet13_after_cancel")
        self._prepared_token = None

    def submit_prepared(self, prepared: PreparedPurchase) -> PurchaseReceipt:
        self._assert_prepared(prepared)
        total = sum(item.amount_yen for item in prepared.intents)
        try:
            self._validate_summary(
                self._body_text(), self._race_stub(prepared), prepared.intents, total
            )
        except BrowserContractError as exc:
            # まだ送信していないので画面を保存して安全に停止できる。
            raise self._contract_error(str(exc), "submit_summary_mismatch") from exc
        self._submitted = True
        try:
            self._click_text(("投票",))
            self._click_text(("はい",))
            self._page.wait_for_load_state("domcontentloaded")
            text = self._body_text()
        except Exception as exc:
            raise BrowserContractError(
                "送信開始後に結果を確認できません。再送せず投票内容照会が必要です"
            ) from exc
        receipt = re.search(r"受付番号\s*[:：]?\s*([0-9]{1,8})", text)
        amount = re.search(r"受付金額\s*[:：]?\s*([0-9,]+)\s*円", text)
        if not receipt or not amount:
            return PurchaseReceipt(
                accepted=False,
                unknown=True,
                receipt_number=None,
                accepted_at=None,
                amount_yen=total,
                summary="受付番号または受付金額を確認できません",
                raw_reference=None,
            )
        received_amount = int(amount.group(1).replace(",", ""))
        if received_amount != total:
            return PurchaseReceipt(
                accepted=False,
                unknown=True,
                receipt_number=receipt.group(1),
                accepted_at=dt.datetime.now(JST),
                amount_yen=received_amount,
                summary="受付金額が購入計画と不一致",
                raw_reference=receipt.group(1),
            )
        self._prepared_token = None
        return PurchaseReceipt(
            accepted=True,
            unknown=False,
            receipt_number=receipt.group(1),
            accepted_at=dt.datetime.now(JST),
            amount_yen=received_amount,
            # 画面全文には会員情報・残高が含まれ得るため永続化しない。
            summary=f"受付番号={receipt.group(1)} 受付金額={received_amount}円",
            raw_reference=receipt.group(1),
        )

    def _assert_prepared(self, prepared: PreparedPurchase) -> None:
        if self._prepared_token != prepared.token:
            raise BrowserContractError("準備済み購入のトークンが一致しません")

    @staticmethod
    def _race_stub(prepared: PreparedPurchase) -> RaceInfo:
        # 確認画面の再検証に必要なのは race_id から得られるレース番号だけ。
        return RaceInfo(
            race_id=prepared.race_id,
            rt_key="",
            venue_code="",
            race_number=int(prepared.race_id[-2:]),
            post_datetime=dt.datetime.now(JST),
        )

    def close(self) -> None:
        for target in (self._context, self._browser):
            if target is not None:
                try:
                    target.close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
        self._page = self._context = self._browser = self._playwright = None
