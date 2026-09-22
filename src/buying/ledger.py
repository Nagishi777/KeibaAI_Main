"""SQLiteによる購入状態・冪等性台帳。"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from src.buying.domain import BetIntent, PurchaseState

TERMINAL_OR_RESERVED = {
    PurchaseState.VALIDATED.value,
    PurchaseState.SUBMITTING.value,
    PurchaseState.ACCEPTED.value,
    PurchaseState.UNKNOWN.value,
    PurchaseState.MANUAL_REVIEW.value,
    PurchaseState.DRY_RUN.value,
}

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    PurchaseState.PLANNED.value: {
        PurchaseState.VALIDATED.value,
        PurchaseState.REJECTED.value,
        PurchaseState.EXPIRED.value,
    },
    PurchaseState.VALIDATED.value: {
        PurchaseState.SUBMITTING.value,
        PurchaseState.REJECTED.value,
        PurchaseState.EXPIRED.value,
        PurchaseState.DRY_RUN.value,
    },
    PurchaseState.SUBMITTING.value: {
        PurchaseState.ACCEPTED.value,
        PurchaseState.UNKNOWN.value,
        PurchaseState.MANUAL_REVIEW.value,
    },
    PurchaseState.UNKNOWN.value: {
        PurchaseState.ACCEPTED.value,
        PurchaseState.NOT_FOUND.value,
        PurchaseState.MANUAL_REVIEW.value,
    },
    PurchaseState.NOT_FOUND.value: {PurchaseState.MANUAL_REVIEW.value},
}


@dataclass(frozen=True)
class LedgerEntry:
    idempotency_key: str
    natural_key: str
    state: str
    race_id: str
    amount_yen: int
    receipt_number: str | None


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS purchases (
                    idempotency_key TEXT PRIMARY KEY,
                    natural_key TEXT NOT NULL UNIQUE,
                    account_alias TEXT NOT NULL,
                    purchase_date TEXT NOT NULL,
                    decision_run_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    race_id TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    selection_json TEXT NOT NULL,
                    amount_yen INTEGER NOT NULL,
                    strategy_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    receipt_number TEXT,
                    receipt_summary TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_purchases_race ON purchases(race_id);
                CREATE INDEX IF NOT EXISTS idx_purchases_state ON purchases(state);
                """
            )
            columns = {
                str(row["name"])
                for row in connection.execute("PRAGMA table_info(purchases)").fetchall()
            }
            if "purchase_date" not in columns:
                connection.execute(
                    "ALTER TABLE purchases ADD COLUMN purchase_date TEXT NOT NULL DEFAULT ''"
                )

    @staticmethod
    def keys(
        account_alias: str, intent: BetIntent, decision_run_id: str
    ) -> tuple[str, str]:
        common = {
            "account_alias": account_alias,
            "race_id": intent.race_id,
            "bet_type": intent.bet_type.value,
            "selection": intent.normalized_selection(),
            "amount_yen": intent.amount_yen,
            "strategy_id": intent.strategy_id,
        }
        natural_raw = json.dumps(common, sort_keys=True, separators=(",", ":"))
        natural_key = hashlib.sha256(natural_raw.encode("utf-8")).hexdigest()
        idempotency_key = hashlib.sha256(
            f"{natural_key}:{decision_run_id}".encode("utf-8")
        ).hexdigest()
        return idempotency_key, natural_key

    def reserve(
        self,
        account_alias: str,
        intent: BetIntent,
        decision_run_id: str,
        input_hash: str,
        purchase_date: dt.date | None = None,
    ) -> tuple[LedgerEntry, bool]:
        idempotency_key, natural_key = self.keys(account_alias, intent, decision_run_id)
        now = dt.datetime.now(dt.timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM purchases WHERE natural_key = ?", (natural_key,)
            ).fetchone()
            if existing is not None:
                return self._entry(existing), False
            connection.execute(
                """
                INSERT INTO purchases (
                    idempotency_key, natural_key, account_alias, purchase_date, decision_run_id,
                    input_hash, race_id, bet_type, selection_json, amount_yen,
                    strategy_id, state, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    idempotency_key,
                    natural_key,
                    account_alias,
                    (purchase_date or dt.datetime.now().date()).isoformat(),
                    decision_run_id,
                    input_hash,
                    intent.race_id,
                    intent.bet_type.value,
                    json.dumps(intent.normalized_selection()),
                    intent.amount_yen,
                    intent.strategy_id,
                    PurchaseState.PLANNED.value,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM purchases WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            return self._entry(row), True

    def transition(
        self,
        idempotency_key: str,
        new_state: PurchaseState,
        *,
        receipt_number: str | None = None,
        receipt_summary: str | None = None,
        error: str | None = None,
    ) -> LedgerEntry:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM purchases WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if row is None:
                raise KeyError(f"台帳に購入がありません: {idempotency_key}")
            old_state = str(row["state"])
            if new_state.value != old_state and new_state.value not in ALLOWED_TRANSITIONS.get(
                old_state, set()
            ):
                raise ValueError(f"不正な状態遷移です: {old_state} -> {new_state.value}")
            connection.execute(
                """
                UPDATE purchases SET state = ?, receipt_number = COALESCE(?, receipt_number),
                    receipt_summary = COALESCE(?, receipt_summary), error = ?, updated_at = ?
                WHERE idempotency_key = ?
                """,
                (
                    new_state.value,
                    receipt_number,
                    receipt_summary,
                    error,
                    dt.datetime.now(dt.timezone.utc).isoformat(),
                    idempotency_key,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM purchases WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            return self._entry(updated)

    def committed_amount(self, target_date: dt.date) -> int:
        states = (
            PurchaseState.SUBMITTING.value,
            PurchaseState.ACCEPTED.value,
            PurchaseState.UNKNOWN.value,
            PurchaseState.MANUAL_REVIEW.value,
        )
        placeholders = ",".join("?" for _ in states)
        with self._connect() as connection:
            day_row = connection.execute(
                f"SELECT COALESCE(SUM(amount_yen), 0) AS amount FROM purchases "
                f"WHERE purchase_date = ? AND state IN ({placeholders})",
                (target_date.isoformat(), *states),
            ).fetchone()
            return int(day_row["amount"])

    def unknown_entries(self) -> list[LedgerEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM purchases WHERE state IN (?, ?) ORDER BY created_at",
                (PurchaseState.SUBMITTING.value, PurchaseState.UNKNOWN.value),
            ).fetchall()
        return [self._entry(row) for row in rows]

    def find_natural(
        self, account_alias: str, intent: BetIntent, decision_run_id: str
    ) -> LedgerEntry | None:
        _, natural_key = self.keys(account_alias, intent, decision_run_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM purchases WHERE natural_key = ?", (natural_key,)
            ).fetchone()
        return None if row is None else self._entry(row)

    def get(self, idempotency_key: str) -> LedgerEntry:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM purchases WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
        if row is None:
            raise KeyError(f"台帳に購入がありません: {idempotency_key}")
        return self._entry(row)

    @staticmethod
    def _entry(row: sqlite3.Row) -> LedgerEntry:
        return LedgerEntry(
            idempotency_key=str(row["idempotency_key"]),
            natural_key=str(row["natural_key"]),
            state=str(row["state"]),
            race_id=str(row["race_id"]),
            amount_yen=int(row["amount_yen"]),
            receipt_number=row["receipt_number"],
        )
