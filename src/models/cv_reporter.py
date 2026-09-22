"""
クロスバリデーション結果レポートモジュール

CVフォールド結果のログ出力ユーティリティ。
静的関数のみで構成されており、外部依存はなし。

ログ出力は学習本体に対する補助機能のため、辞書キーの欠損や型のずれで
学習全体が止まらないよう、防御的アクセス（_num / _fnum）と try/except で保護する。
"""
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def _num(d: dict, key: str) -> float:
    """辞書から数値を取り出す。欠損・非数値は nan を返す（KeyError/TypeError 防御）。"""
    v = d.get(key, float('nan'))
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _isnan(x: float) -> bool:
    """nan 判定。非数値でも例外を投げない。"""
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True


def _fnum(d: dict, key: str, fmt: str, suffix: str = "") -> str:
    """辞書の数値を fmt で整形する。欠損・nan は 'N/A' を返す。"""
    v = _num(d, key)
    if _isnan(v):
        return "  N/A "
    return format(v, fmt) + suffix


def _s(d: dict, key: str) -> Any:
    """辞書から文字列系の値を取り出す。欠損は '?' を返す。"""
    return d.get(key, '?')


def log_cv_fold_table(fold_results: list, cv_result: dict) -> None:
    """CVフォールド結果をテーブル形式でログ出力する。

    Args:
        fold_results: フォールドごとの評価結果リスト。各要素に
            fold / train_start / train_end / test_start / test_end /
            auc / hit_rate / recovery_rate を含む。
        cv_result: CV全体のサマリ辞書。mean_auc / std_auc /
            mean_hit_rate / std_hit_rate / mean_recovery_rate /
            std_recovery_rate を含む。
    """
    try:
        header = f"{'Fold':>4}  {'Train':>12}  {'Test':>12}  {'AUC':>6}  {'的中率':>7}  {'回収率':>7}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in fold_results:
            logger.info(
                f"{_s(r, 'fold'):>4}  {_s(r, 'train_start')}〜{_s(r, 'train_end')}  "
                f"{_s(r, 'test_start')}〜{_s(r, 'test_end')}  "
                f"{_fnum(r, 'auc', '.4f'):>6}  "
                f"{_fnum(r, 'hit_rate', '.1f', '%'):>7}  "
                f"{_fnum(r, 'recovery_rate', '.1f', '%'):>7}"
            )
        summary_msg = (
            f"平均: AUC={_fnum(cv_result, 'mean_auc', '.4f')} "
            f"(+/-{_fnum(cv_result, 'std_auc', '.4f')})"
        )
        # 「値が有効なフォールドが1件以上あったか」で表示判定する。
        # `if mean_hit_rate:` だと的中率0.0%（全外れ）を「データなし」と誤判定するため
        # 有効フォールド有無で判定する。
        has_hit = any(not _isnan(_num(r, 'hit_rate')) for r in fold_results)
        has_rr = any(not _isnan(_num(r, 'recovery_rate')) for r in fold_results)
        if has_hit:
            summary_msg += (
                f" | 的中率={_fnum(cv_result, 'mean_hit_rate', '.1f', '%')}"
                f" (+/-{_fnum(cv_result, 'std_hit_rate', '.1f', '%')})"
            )
        if has_rr:
            summary_msg += (
                f" | 回収率={_fnum(cv_result, 'mean_recovery_rate', '.1f', '%')}"
                f" (+/-{_fnum(cv_result, 'std_recovery_rate', '.1f', '%')})"
            )
        logger.info(summary_msg)
    except Exception as e:
        # ログ出力は補助機能。整形に失敗しても学習本体は止めない。
        logger.warning(f"CVフォールド結果のログ出力に失敗しました（スキップ）: {e}")


def log_ranking_cv_fold_table(fold_results: list, cv_result: dict) -> None:
    """ランキングCVフォールド結果をテーブル形式でログ出力する。

    Args:
        fold_results: フォールドごとの評価結果リスト。各要素に
            fold / train_start / train_end / test_start / test_end /
            hit_rate / recovery_rate を含む。
        cv_result: CV全体のサマリ辞書。mean_hit_rate / std_hit_rate /
            mean_recovery_rate / std_recovery_rate を含む。
    """
    try:
        header = f"{'Fold':>4}  {'Train':>12}  {'Test':>12}  {'top1的中率':>10}  {'回収率':>7}"
        logger.info(header)
        logger.info("-" * len(header))
        for r in fold_results:
            logger.info(
                f"{_s(r, 'fold'):>4}  {_s(r, 'train_start')}〜{_s(r, 'train_end')}  "
                f"{_s(r, 'test_start')}〜{_s(r, 'test_end')}  "
                f"{_fnum(r, 'hit_rate', '.1f', '%'):>10}  "
                f"{_fnum(r, 'recovery_rate', '.1f', '%'):>7}"
            )
        summary_msg = (
            f"平均: top1的中率={_fnum(cv_result, 'mean_hit_rate', '.1f', '%')}"
            f" (+/-{_fnum(cv_result, 'std_hit_rate', '.1f', '%')})"
        )
        has_rr = any(not _isnan(_num(r, 'recovery_rate')) for r in fold_results)
        if has_rr:
            summary_msg += (
                f" | 回収率={_fnum(cv_result, 'mean_recovery_rate', '.1f', '%')}"
                f" (+/-{_fnum(cv_result, 'std_recovery_rate', '.1f', '%')})"
            )
        logger.info(summary_msg)
    except Exception as e:
        logger.warning(f"ランキングCVフォールド結果のログ出力に失敗しました（スキップ）: {e}")
