"""``src/buying/.env`` の読込と厳格な設定検証。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.buying.domain import BuyingMode, Credentials


class ConfigurationError(ValueError):
    """安全に実行できない設定。"""


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigurationError(f"{path}:{line_number}: KEY=VALUE 形式ではありません")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if not key:
            raise ConfigurationError(f"{path}:{line_number}: キーが空です")
        values[key] = value
    return values


def _get(values: dict[str, str], key: str, default: str = "") -> str:
    return os.environ.get(key, values.get(key, default)).strip()


def _bool(values: dict[str, str], key: str, default: bool) -> bool:
    raw = _get(values, key, "true" if default else "false").lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{key} は true/false で指定してください: {raw!r}")


def _int(values: dict[str, str], key: str, default: int, *, minimum: int = 0) -> int:
    raw = _get(values, key, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{key} は整数で指定してください: {raw!r}") from exc
    if value < minimum:
        raise ConfigurationError(f"{key} は {minimum} 以上で指定してください: {value}")
    return value


@dataclass(frozen=True)
class BuyingSettings:
    mode: BuyingMode
    kill_switch: bool
    headless: bool
    account_alias: str
    credentials: Credentials | None
    decision_lead_seconds: int
    submit_deadline_lead_seconds: int
    max_clock_skew_seconds: int
    max_odds_age_seconds: int
    min_bet_yen: int
    max_bet_yen: int
    max_race_yen: int
    max_day_yen: int
    max_session_yen: int
    max_bets_per_race: int
    allowed_bet_types: frozenset[str]
    schedule_dir: Path
    odds_dir: Path
    data_dir: Path
    login_url: str
    poll_interval_seconds: int

    @property
    def ledger_path(self) -> Path:
        return self.data_dir / "buying_ledger.sqlite3"

    @property
    def log_dir(self) -> Path:
        return self.data_dir / "logs"


def load_settings(
    env_path: Path | None = None,
    *,
    require_credentials: bool | None = None,
) -> BuyingSettings:
    env_path = env_path or Path(__file__).with_name(".env")
    values = _read_dotenv(Path(env_path))
    raw_mode = _get(values, "BUYING_MODE", BuyingMode.DRY_RUN.value)
    try:
        mode = BuyingMode(raw_mode)
    except ValueError as exc:
        raise ConfigurationError("BUYING_MODE は paper/dry-run/live のいずれかです") from exc

    should_require = mode != BuyingMode.PAPER if require_credentials is None else require_credentials
    credential_values = {
        "inet_id": _get(values, "JRA_INET_ID"),
        "subscriber_number": _get(values, "JRA_KANYUSHA_NO"),
        "pin": _get(values, "JRA_PIN"),
        "pars_number": _get(values, "JRA_PARS_NO"),
    }
    invalid = [
        name for name, value in credential_values.items()
        if not value or value.lower() in {"replace_me", "changeme"}
    ]
    if should_require and invalid:
        raise ConfigurationError(
            "認証情報が未設定です: " + ", ".join(invalid) + f"（{env_path} を確認してください）"
        )
    credentials = Credentials(**credential_values) if not invalid else None

    settings = BuyingSettings(
        mode=mode,
        kill_switch=_bool(values, "BUYING_KILL_SWITCH", True),
        headless=_bool(values, "BUYING_HEADLESS", False),
        account_alias=_get(values, "BUYING_ACCOUNT_ALIAS", "default"),
        credentials=credentials,
        decision_lead_seconds=_int(values, "BUYING_DECISION_LEAD_SECONDS", 180),
        submit_deadline_lead_seconds=_int(
            values, "BUYING_SUBMIT_DEADLINE_LEAD_SECONDS", 120
        ),
        max_clock_skew_seconds=_int(values, "BUYING_MAX_CLOCK_SKEW_SECONDS", 2),
        max_odds_age_seconds=_int(values, "BUYING_MAX_ODDS_AGE_SECONDS", 180),
        min_bet_yen=_int(values, "BUYING_MIN_BET_YEN", 100, minimum=100),
        max_bet_yen=_int(values, "BUYING_MAX_BET_YEN", 100, minimum=100),
        max_race_yen=_int(values, "BUYING_MAX_RACE_YEN", 100, minimum=100),
        max_day_yen=_int(values, "BUYING_MAX_DAY_YEN", 100, minimum=100),
        max_session_yen=_int(values, "BUYING_MAX_SESSION_YEN", 100, minimum=100),
        max_bets_per_race=_int(values, "BUYING_MAX_BETS_PER_RACE", 1, minimum=1),
        allowed_bet_types=frozenset(
            part.strip() for part in _get(values, "BUYING_ALLOWED_BET_TYPES", "win").split(",")
            if part.strip()
        ),
        schedule_dir=Path(_get(values, "BUYING_SCHEDULE_DIR", "data/processed/schedules")),
        odds_dir=Path(_get(values, "BUYING_ODDS_DIR", "data/processed/realtime_odds")),
        data_dir=Path(_get(values, "BUYING_DATA_DIR", "data/processed/buying")),
        login_url=_get(values, "BUYING_LOGIN_URL", "https://www.ipat.jra.go.jp/"),
        poll_interval_seconds=_int(values, "BUYING_POLL_INTERVAL_SECONDS", 15, minimum=1),
    )
    _validate_limits(settings)
    return settings


def _validate_limits(settings: BuyingSettings) -> None:
    values = (
        settings.min_bet_yen,
        settings.max_bet_yen,
        settings.max_race_yen,
        settings.max_day_yen,
        settings.max_session_yen,
    )
    if any(value % 100 for value in values):
        raise ConfigurationError("購入金額・上限はすべて100円単位で指定してください")
    if settings.min_bet_yen > settings.max_bet_yen:
        raise ConfigurationError("BUYING_MIN_BET_YEN が BUYING_MAX_BET_YEN を超えています")
    if settings.max_bet_yen > settings.max_race_yen:
        raise ConfigurationError("1点上限が1レース上限を超えています")
    if settings.max_race_yen > settings.max_day_yen:
        raise ConfigurationError("1レース上限が1日上限を超えています")
    if settings.submit_deadline_lead_seconds < 60:
        raise ConfigurationError("送信期限は公式締切より安全側の60秒以上にしてください")

