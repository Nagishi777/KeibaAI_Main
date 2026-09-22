"""ブラウザ境界。テストではFake実装へ差し替える。"""

from __future__ import annotations

from typing import Protocol

from src.buying.domain import (
    BetIntent,
    Credentials,
    PreparedPurchase,
    PurchaseReceipt,
    RaceInfo,
)


class BrowserContractError(RuntimeError):
    """即PAT画面が期待した契約と一致しない。liveでは即時停止対象。"""


class IpatClient(Protocol):
    def login(self, credentials: Credentials) -> None: ...

    def get_balance_yen(self) -> int: ...

    def prepare_win_bets(
        self, race: RaceInfo, intents: tuple[BetIntent, ...]
    ) -> PreparedPurchase: ...

    def cancel_prepared(self, prepared: PreparedPurchase) -> None: ...

    def submit_prepared(self, prepared: PreparedPurchase) -> PurchaseReceipt: ...

    def close(self) -> None: ...

