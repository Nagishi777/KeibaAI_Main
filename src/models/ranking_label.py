"""
ランキング関連度ラベル生成モジュール

LambdaRank 学習に使う関連度スコア（非負整数ラベル）を、finish_position
（および odds_win）からベクトル演算で生成する。

trainer.py / optimizer.py の複数箇所で個別実装されていたラベル生成を
単一の実装に集約し、CV と本学習でラベル定義がずれる問題を防ぐ。
"""
import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# デフォルトの label_gain と整合する既定関連度スコア
_DEFAULT_TOP1 = 2
_DEFAULT_TOP3 = 1
_DEFAULT_OTHER = 0


def compute_dynamic_odds_thresholds(train_df: pd.DataFrame) -> Dict[str, float]:
    """train_df のみからオッズ閾値を計算する（リーク防止）。

    Args:
        train_df: 'odds_win' カラムを持つ学習データ。

    Returns:
        Dict[str, float]: high_win / mid_win / high_place の閾値辞書。
    """
    return {
        'mid_win': float(train_df['odds_win'].quantile(0.25)),
        'high_win': float(train_df['odds_win'].quantile(0.75)),
        'high_place': float(train_df['odds_win'].quantile(0.90)),
    }


def make_relevance_labels(
    df: pd.DataFrame,
    cfg: dict,
    thresholds: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """finish_position（+odds_win）から関連度ラベルをベクトル生成する。

    scheme に応じて 2 種類のラベル体系を返す:
        - 'positional'（デフォルト）: 1着=top1, 3着以内=top3, その他=other
        - 'odds_weighted' / 'odds_weighted_dynamic':
              1着かつ高配当=4 / 1着かつ中オッズ=3 / 1着=2 /
              2着=top2 / 3着以内かつ高配当=2 / 3着以内=1 / その他=0

    Args:
        df: 'finish_position' カラムを持つ DataFrame（odds_weighted 時は 'odds_win' も必須）。
        cfg: ranking_label 設定辞書。
        thresholds: odds_weighted_dynamic 用に train のみで計算した閾値。
            省略時は cfg の固定閾値を使用する。

    Returns:
        np.ndarray: 各行の関連度ラベル（int）。
    """
    pos = df['finish_position'].to_numpy(dtype=float)
    scheme = cfg.get('scheme', 'positional')

    # finish_position には出走取消(-1)・発走除外(-2)・失格/競走中止(-3) を表す負値が入る。
    # `pos <= 3` だけだとこれらを「3着以内」と誤判定して高い関連度を与えてしまうため、
    # 3着以内の判定には必ずこのマスクを使う（下限も確認する）。
    in_top3 = (pos >= 1) & (pos <= 3)

    if scheme in ('odds_weighted', 'odds_weighted_dynamic') and 'odds_win' in df.columns:
        odds = df['odds_win'].to_numpy(dtype=float)
        if thresholds is not None:
            thr_high_win = thresholds['high_win']
            thr_mid_win = thresholds['mid_win']
            thr_high_place = thresholds['high_place']
        else:
            thr_high_win = float(cfg.get('odds_threshold_high_win', 10.0))
            thr_mid_win = float(cfg.get('odds_threshold_mid_win', 4.0))
            thr_high_place = float(cfg.get('odds_threshold_high_place', 15.0))
        top2_score = int(cfg.get('top2', 1))

        # 1着なのに odds_win が NaN の馬は、odds >= 閾値 が全て False になり
        # 最低評価のラベル2（本命的中相当）に落ちる。穴馬的中（本来ラベル3〜4）が
        # 静かに過小評価されるため、件数を必ず可視化する。
        one_win_nan = (pos == 1) & np.isnan(odds)
        n_nan = int(one_win_nan.sum())
        if n_nan > 0:
            msg = (
                f"[ranking_label] scheme={scheme}: odds_win が NaN の1着馬 {n_nan} 件を"
                f"ラベル2に評価しました（穴馬的中の過小評価の可能性）"
            )
            if cfg.get('strict_odds', False):
                # CLAUDE.md「欠損は補完せず処理を止める」に従う厳格モード
                raise ValueError(msg + " [strict_odds=True のため停止]")
            logger.warning(msg)

        # 上から順にマッチ（np.select は最初に True になった条件を採用）
        conditions = [
            np.isnan(pos),
            (pos == 1) & (odds >= thr_high_win),
            (pos == 1) & (odds >= thr_mid_win),
            pos == 1,
            pos == 2,
            in_top3 & (odds >= thr_high_place),
            in_top3,
        ]
        choices = [0, 4, 3, 2, top2_score, 2, 1]
        return np.select(conditions, choices, default=0).astype(int)

    top1 = int(cfg.get('top1', _DEFAULT_TOP1))
    top3 = int(cfg.get('top3', _DEFAULT_TOP3))
    other = int(cfg.get('other', _DEFAULT_OTHER))
    conditions = [np.isnan(pos), pos == 1, in_top3]
    choices = [other, top1, top3]
    return np.select(conditions, choices, default=other).astype(int)
