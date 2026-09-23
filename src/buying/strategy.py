"""``src/simulator`` の購入推奨を購入指示へ変換するアダプタ。"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.buying.domain import BetIntent, BetType
from src.simulator.features import INC_SHARE_COL, ODDS_COL, POOL_COL

STRATEGY_ID = "simulator_pool_filter_v1"


class StrategyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StrategyResult:
    intents: tuple[BetIntent, ...]
    decision_run_id: str
    input_hash: str
    summary: Any | None = None


class SimulatorStrategy:
    """シミュレータを直接実行、またはその推奨CSVを読み込む。"""

    def __init__(self, *, bet_unit_yen: int = 100) -> None:
        self.bet_unit_yen = bet_unit_yen

    def run(
        self,
        *,
        target_date: dt.date,
        simulator_config_path: Path,
        realtime_dir: Path,
        model_filename: str | None = None,
        min_ev: float | None = None,
        fetch: bool = False,
        headless: bool = True,
        output_dir: Path | None = None,
    ) -> StrategyResult:
        try:
            from src.cli_common import build_config
            from src.simulator.artifacts import DEFAULT_MODEL_FILENAME
            from src.simulator.predict_today import run_predict_today
        except (ImportError, ModuleNotFoundError) as exc:
            raise StrategyError(
                "src/simulator の依存関係を読み込めません。requirement.txt を導入し、"
                "シミュレータ単体が実行できることを確認してください。"
            ) from exc
        config = build_config(simulator_config_path)
        summary, frame = run_predict_today(
            config,
            target_date=target_date,
            realtime_dir=realtime_dir,
            model_filename=model_filename or DEFAULT_MODEL_FILENAME,
            min_ev=min_ev,
            fetch=fetch,
            headless=headless,
            output_dir=output_dir,
        )
        return self.from_frame(frame, target_date=target_date, summary=summary)

    def from_csv(self, path: Path, *, target_date: dt.date) -> StrategyResult:
        if not Path(path).exists():
            raise FileNotFoundError(f"シミュレータ推奨CSVが見つかりません: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype={"race_id": str})
        return self.from_frame(frame, target_date=target_date)

    def from_frame(
        self,
        frame: pd.DataFrame,
        *,
        target_date: dt.date,
        summary: Any | None = None,
    ) -> StrategyResult:
        required = {"race_id", "horse_number", "bet_amount"}
        missing = required - set(frame.columns)
        if missing:
            raise StrategyError(f"シミュレータ出力に必要な列がありません: {sorted(missing)}")
        selected = frame.copy()
        if "skip_reason" in selected.columns:
            selected = selected[selected["skip_reason"].fillna("").astype(str) == ""]
        selected["bet_amount"] = pd.to_numeric(selected["bet_amount"], errors="coerce")
        selected["horse_number"] = pd.to_numeric(selected["horse_number"], errors="coerce")
        if selected[["bet_amount", "horse_number"]].isna().any().any():
            raise StrategyError("シミュレータ出力の馬番または賭け金に不正値があります")

        intents: list[BetIntent] = []
        for row in selected.to_dict("records"):
            raw_amount = float(row["bet_amount"])
            amount = int(raw_amount // self.bet_unit_yen) * self.bet_unit_yen
            if amount < self.bet_unit_yen:
                continue
            horse = int(row["horse_number"])
            metadata = {
                key: row[key]
                for key in ("pred_proba", ODDS_COL, "ev", POOL_COL, INC_SHARE_COL)
                if key in row and not pd.isna(row[key])
            }
            intents.append(
                BetIntent(
                    race_id=str(row["race_id"]).zfill(12),
                    bet_type=BetType.WIN,
                    selection=(horse,),
                    amount_yen=amount,
                    strategy_id=STRATEGY_ID,
                    reason_code="simulator_selected",
                    expected_odds=(
                        float(row[ODDS_COL])
                        if ODDS_COL in row and not pd.isna(row[ODDS_COL])
                        else None
                    ),
                    metadata=metadata,
                )
            )
        intents.sort(key=lambda item: (item.race_id, item.selection, item.amount_yen))
        payload = [intent.as_public_dict() for intent in intents]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        input_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        decision_run_id = hashlib.sha256(
            f"{target_date.isoformat()}:{STRATEGY_ID}:{input_hash}".encode("utf-8")
        ).hexdigest()[:24]
        return StrategyResult(tuple(intents), decision_run_id, input_hash, summary)

