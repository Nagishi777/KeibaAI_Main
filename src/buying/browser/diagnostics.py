"""セレクタ不一致時の画面ダンプ。

即PATの画面変更でロケータが外れたとき、何が表示されていたのかを後から
確認できるようにする。会員情報・残高・認証情報はダンプ前にマスクする。
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_DUMP_DIR = Path("data/processed/buying/diagnostics")

_MASK = "***MASKED***"

# 画面に出る会員情報・金額。桁数を残すと特定に使えるため全体を置き換える。
_TEXT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(INET[-\s]?ID\s*[:：]?\s*)[0-9A-Za-z]+", r"\1" + _MASK),
    (r"(加入者番号\s*[:：]?\s*)[0-9]+", r"\1" + _MASK),
    (r"(暗証番号\s*[:：]?\s*)[0-9]+", r"\1" + _MASK),
    (r"(P-?ARS番号\s*[:：]?\s*)[0-9]+", r"\1" + _MASK),
    (r"(購入限度額\s*[:：]?\s*)[0-9,]+(\s*円)", r"\1" + _MASK + r"\2"),
    (r"(残高\s*[:：]?\s*)[0-9,]+(\s*円)", r"\1" + _MASK + r"\2"),
    (r"(ご利用可能額\s*[:：]?\s*)[0-9,]+(\s*円)", r"\1" + _MASK + r"\2"),
)

# input タグ全体。value属性の前後どちらに type があっても判定できるようにする。
_INPUT_TAG_PATTERN = re.compile(r"<input\b[^>]*>", re.IGNORECASE)

_VALUE_ATTR_PATTERN = re.compile(
    r'(\bvalue\s*=\s*)(["\'])(.*?)\2', re.IGNORECASE | re.DOTALL
)

# submit/button/reset のvalueはボタン名そのもので、認証情報を含まない。
# セレクタ調査に必要な情報なので残す。
_BUTTON_TYPE_PATTERN = re.compile(
    r'\btype\s*=\s*["\']?(?:submit|button|reset|image)\b', re.IGNORECASE
)


def _mask_text(text: str, secrets: Sequence[str]) -> str:
    masked = text
    # 既知の認証情報は完全一致で先に潰す（画面ラベルに依存しないため確実）。
    for secret in secrets:
        if secret:
            masked = masked.replace(secret, _MASK)
    for pattern, replacement in _TEXT_PATTERNS:
        masked = re.sub(pattern, replacement, masked)
    return masked


def _mask_input_tag(match: re.Match[str]) -> str:
    tag = match.group(0)
    if _BUTTON_TYPE_PATTERN.search(tag):
        return tag
    return _VALUE_ATTR_PATTERN.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{_MASK}{m.group(2)}", tag
    )


def _mask_html(html: str, secrets: Sequence[str]) -> str:
    masked = _INPUT_TAG_PATTERN.sub(_mask_input_tag, html)
    return _mask_text(masked, secrets)


_CHARSET_PATTERN = re.compile(r"<meta[^>]*charset[^>]*>", re.IGNORECASE)


def _force_utf8_charset(html: str) -> str:
    """元ページのcharset宣言をUTF-8へ直す。

    ダンプはUTF-8で書き出すため、即PATのShift_JIS宣言が残っているとブラウザで
    開いたときに文字化けし、セレクタ調査が成立しない。
    """
    if _CHARSET_PATTERN.search(html):
        return _CHARSET_PATTERN.sub('<meta charset="utf-8">', html, count=1)
    return html


def _safe(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_") or "dump"


class ScreenDumper:
    """ページとフレームの内容をマスクして保存する。"""

    def __init__(self, dump_dir: Path | None = None) -> None:
        self.dump_dir = Path(dump_dir) if dump_dir is not None else DEFAULT_DUMP_DIR
        self._secrets: list[str] = []

    def register_secrets(self, *values: str | None) -> None:
        """マスク対象の認証情報を登録する（ダンプ内容には残さない）。"""
        for value in values:
            if value and value not in self._secrets:
                self._secrets.append(value)

    def dump(self, page: Any, label: str, *, scopes: Iterable[Any] = ()) -> Path | None:
        """現在の画面を保存し、保存先を返す。失敗しても例外は送出しない。"""
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.dump_dir / f"{stamp}_{_safe(label)}"
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.exception("診断ダンプの保存先を作成できません: %s", target)
            return None

        self._write_screenshot(page, target)
        self._write_frames(page, scopes, target)
        logger.warning("画面ダンプを保存しました: %s", target)
        return target

    def _write_screenshot(self, page: Any, target: Path) -> None:
        try:
            page.screenshot(path=str(target / "screenshot.png"), full_page=True)
        except Exception:
            logger.exception("スクリーンショットを取得できません")

    def _write_frames(self, page: Any, scopes: Iterable[Any], target: Path) -> None:
        seen: list[Any] = []
        for scope in (page, *scopes):
            if any(scope is item for item in seen):
                continue
            seen.append(scope)
            index = len(seen) - 1
            name = "main" if index == 0 else f"frame{index}"
            self._write_frame(scope, target, name)

    def _write_frame(self, scope: Any, target: Path, name: str) -> None:
        try:
            html = scope.content()
        except Exception:
            try:
                html = scope.locator("html").inner_html()
            except Exception:
                return
        try:
            url = scope.url
        except Exception:
            url = "(unknown)"
        header = f"<!-- url: {_mask_text(str(url), self._secrets)} -->\n"
        try:
            (target / f"{name}.html").write_text(
                header + _force_utf8_charset(_mask_html(html, self._secrets)),
                encoding="utf-8",
            )
        except OSError:
            logger.exception("HTMLダンプを書き出せません: %s", name)

        self._write_candidates(scope, target, name)

    def _write_candidates(self, scope: Any, target: Path, name: str) -> None:
        """クリック候補の要素一覧。セレクタ修正の当たりを付けるために出す。"""
        lines: list[str] = []
        for selector in ("button", "a", "input", "[role=button]", "[onclick]"):
            try:
                locator = scope.locator(selector)
                count = locator.count()
            except Exception:
                continue
            lines.append(f"## {selector} ({count}件)")
            for i in range(min(count, 60)):
                lines.append(f"  [{i}] {self._describe(locator.nth(i))}")
        if not lines:
            return
        try:
            (target / f"{name}.candidates.txt").write_text(
                _mask_text("\n".join(lines), self._secrets), encoding="utf-8"
            )
        except OSError:
            logger.exception("候補一覧を書き出せません: %s", name)

    def _describe(self, element: Any) -> str:
        parts: list[str] = []
        for attr in ("name", "id", "type", "class", "aria-label", "title"):
            try:
                value = element.get_attribute(attr)
            except Exception:
                continue
            if value:
                parts.append(f"{attr}={value!r}")
        try:
            text = (element.inner_text() or "").strip()
        except Exception:
            text = ""
        if not text:
            try:
                text = (element.get_attribute("value") or "").strip()
            except Exception:
                text = ""
        if text:
            parts.append(f"text={text[:60]!r}")
        return " ".join(parts) or "(属性なし)"
