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
from src.simulator.features import DRIFT_COL, POOL_COL

STRATEGY_ID = "anaba_steam_move_v1"
PLACE_STRATEGY_ID = "anaba_place_v1"

# 券種ごとの戦略ID（台帳の自然キーに入り、単勝と複勝の購入を区別する）
STRATEGY_IDS = {BetType.WIN: STRATEGY_ID, BetType.PLACE: PLACE_STRATEGY_ID}


class StrategyError(RuntimeError):
    pass


@dataclass(frozen=True)
class StrategyResult:
    intents: tuple[BetIntent, ...]
    decision_run_id: str
    input_hash: str
    summary: Any | None = None


class SimulatorStrategy:
    """シミュレータを直接実行、またはその推奨CSVを読み込む。

    ``bet_type`` で単勝（既定）か複勝かを選ぶ。複勝の推奨は
    :mod:`src.simulator.predict_place` が出力する。
    """

    def __init__(self, *, bet_unit_yen: int = 100, bet_type: BetType = BetType.WIN) -> None:
        if bet_type not in STRATEGY_IDS:
            raise StrategyError(f"未対応の券種です: {bet_type}")
        self.bet_unit_yen = bet_unit_yen
        self.bet_type = bet_type
        self.strategy_id = STRATEGY_IDS[bet_type]

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
            from src.simulator.predict_place import run_predict_place_today
            from src.simulator.predict_today import run_predict_today
        except (ImportError, ModuleNotFoundError) as exc:
            raise StrategyError(
                "src/simulator の依存関係を読み込めません。requirement.txt を導入し、"
                "シミュレータ単体が実行できることを確認してください。"
            ) from exc
        config = build_config(simulator_config_path)
        if self.bet_type == BetType.PLACE:
            if fetch:
                raise StrategyError("複勝は --fetch に対応していません（取得済みのオッズから予測します）")
            summary, frame = run_predict_place_today(
                config,
                target_date=target_date,
                realtime_dir=realtime_dir,
                model_subdir=model_filename,
                min_ev=min_ev,
                output_dir=output_dir,
            )
            return self.from_frame(frame, target_date=target_date, summary=summary)
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
        required = {"race_id", "horse_number"}
        missing = required - set(frame.columns)
        if missing:
            raise StrategyError(f"シミュレータ出力に必要な列がありません: {sorted(missing)}")
        selected = frame.copy()
        if "skip_reason" in selected.columns:
            selected = selected[selected["skip_reason"].fillna("").astype(str) == ""]
        selected["horse_number"] = pd.to_numeric(selected["horse_number"], errors="coerce")
        if selected["horse_number"].isna().any():
            raise StrategyError("シミュレータ出力の馬番に不正値があります")

        intents: list[BetIntent] = []
        for row in selected.to_dict("records"):
            amount = self.bet_unit_yen  # Flat ベット（1点 = 購入単位）
            horse = int(row["horse_number"])
            metadata = {
                key: row[key]
                for key in (
                    "pred_proba", "odds_used", "ev", POOL_COL, DRIFT_COL,
                    "inc_share_5_late", "decision_snapshot_used",
                    "odds_place_max_late",
                )
                if key in row and not pd.isna(row[key])
            }
            intents.append(
                BetIntent(
                    race_id=str(row["race_id"]).zfill(12),
                    bet_type=self.bet_type,
                    selection=(horse,),
                    amount_yen=amount,
                    strategy_id=self.strategy_id,
                    reason_code="simulator_selected",
                    expected_odds=(
                        float(row["odds_used"])
                        if "odds_used" in row and not pd.isna(row["odds_used"])
                        else None
                    ),
                    metadata=metadata,
                )
            )
        return _build_result(intents, target_date, self.strategy_id, summary)


def _build_result(
    intents: list[BetIntent], target_date: dt.date, strategy_id: str, summary: Any | None
) -> StrategyResult:
    intents = sorted(
        intents, key=lambda item: (item.race_id, item.bet_type.value, item.selection, item.amount_yen)
    )
    payload = [intent.as_public_dict() for intent in intents]
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    input_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    decision_run_id = hashlib.sha256(
        f"{target_date.isoformat()}:{strategy_id}:{input_hash}".encode("utf-8")
    ).hexdigest()[:24]
    return StrategyResult(tuple(intents), decision_run_id, input_hash, summary)


def combine_results(results: list[StrategyResult], *, target_date: dt.date) -> StrategyResult:
    """単勝・複勝の購入指示を1回の購入処理（ログイン1回）にまとめる。

    台帳の自然キーは券種と戦略IDを含むため、まとめても単勝と複勝の
    二重購入防止は独立に効く。
    """
    if len(results) == 1:
        return results[0]
    intents = [intent for result in results for intent in result.intents]
    strategy_id = "+".join(sorted({intent.strategy_id for intent in intents})) or "none"
    summaries = [result.summary for result in results]
    return _build_result(intents, target_date, strategy_id, summaries)

