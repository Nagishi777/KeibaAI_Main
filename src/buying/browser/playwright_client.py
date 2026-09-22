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
    ) -> None:
        self.login_url = login_url
        self.headless = headless
        self.timeout_ms = timeout_ms
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
            raise BrowserContractError(f"{description} が一意ではありません（{count}件）")
        return locator

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
        raise BrowserContractError(f"入力欄が見つかりません: {labels}")

    def _click_text(self, names: tuple[str, ...], *, exact: bool = True) -> None:
        for scope in self._scopes():
            for name in names:
                by_role = scope.get_by_role("button", name=name, exact=exact)
                if by_role.count() == 1:
                    by_role.click()
                    return
                by_link = scope.get_by_role("link", name=name, exact=exact)
                if by_link.count() == 1:
                    by_link.click()
                    return
                by_text = scope.get_by_text(name, exact=exact)
                if by_text.count() == 1:
                    by_text.click()
                    return
        raise BrowserContractError(f"操作要素が見つからないか一意ではありません: {names}")

    def _body_text(self) -> str:
        texts: list[str] = []
        for scope in self._scopes():
            try:
                texts.append(scope.locator("body").inner_text())
            except Exception:
                continue
        return "\n".join(texts)

    def login(self, credentials: Credentials) -> None:
        self._ensure_started()
        self._page.goto(self.login_url, wait_until="domcontentloaded")
        inet = self._find_input(
            ("INET-ID", "INET ID"),
            ('input[name*="inet" i]', 'input[id*="inet" i]'),
        )
        inet.fill(credentials.inet_id)
        self._click_text(("ログイン", "送信"))
        self._page.wait_for_load_state("domcontentloaded")

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
        self._click_text(("ログイン", "送信"))
        self._page.wait_for_load_state("domcontentloaded")
        text = self._body_text()
        if not any(word in text for word in ("投票", "購入限度額", "メニュー")):
            raise BrowserContractError("ログイン後のメニューを確認できません")

    def get_balance_yen(self) -> int:
        text = self._body_text()
        match = re.search(r"(?:購入限度額|残高)\s*[:：]?\s*([0-9,]+)\s*円", text)
        if not match:
            raise BrowserContractError("購入限度額を画面から取得できません")
        return int(match.group(1).replace(",", ""))

    def prepare_win_bets(
        self, race: RaceInfo, intents: tuple[BetIntent, ...]
    ) -> PreparedPurchase:
        if not intents:
            raise BrowserContractError("空の購入指示は入力できません")
        venue = VENUE_NAMES.get(race.venue_code)
        if venue is None:
            raise BrowserContractError(f"未対応の競馬場コードです: {race.venue_code}")
        self._click_text(("通常投票",))
        self._click_text((venue,))
        self._click_text((f"{race.race_number}R", str(race.race_number)))
        self._click_text(("単勝",))

        for intent in intents:
            horse = intent.selection[0]
            self._click_text((str(horse), f"{horse}番"))
        self._click_text(("金額入力", "次へ"))

        for intent in intents:
            self._fill_amount_for_horse(intent.selection[0], intent.amount_yen)
        self._click_text(("セット",))
        self._click_text(("入力終了",))
        total = sum(item.amount_yen for item in intents)
        self._fill_confirmation_total(total)
        text = self._body_text()
        self._validate_summary(text, race, intents, total)
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
        raise BrowserContractError(f"馬番{horse}の金額入力欄を一意に取得できません")

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
        self._prepared_token = None

    def submit_prepared(self, prepared: PreparedPurchase) -> PurchaseReceipt:
        self._assert_prepared(prepared)
        total = sum(item.amount_yen for item in prepared.intents)
        self._validate_summary(self._body_text(), self._race_stub(prepared), prepared.intents, total)
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
