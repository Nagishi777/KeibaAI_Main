"""
モデル評価・回収率計算モジュール
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)

from src.models.odds_series import BET_ODDS_COLS, PAYOUT_COLS

logger = logging.getLogger(__name__)

# 回収率計算がサポートする馬券種
SUPPORTED_BET_TYPES = ('win', 'place', 'umaren')


class ModelEvaluator:
    """
    モデルの評価と回収率計算を行うクラス
    """

    def __init__(self, config: dict):
        """
        初期化

        Args:
            config: 設定辞書
        """
        self.config = config
        self.bet_unit = config.get('bet_unit', 100)
        self.use_kelly = config.get('use_kelly_criterion', True)
        self.kelly_fraction = config.get('kelly_fraction', 0.5)
        self.max_bet = config.get('max_bet', 10_000)
        self.initial_bankroll = config.get('initial_bankroll', 100000)
        self.min_proba_win = config.get('min_proba_win', 0.0)
        self.min_proba_place = config.get('min_proba_place', 0.0)
        self.min_proba_umaren = config.get('min_proba_umaren', 0.0)
        self.max_bets_per_race_win = config.get('max_bets_per_race_win', 0)
        self.max_bets_per_race_place = config.get('max_bets_per_race_place', 0)
        # 馬連は組み合わせ数が多く、点数上限を設けないと賭け金が発散するため既定 3 点
        self.max_bets_per_race_umaren = config.get('max_bets_per_race_umaren', 3)
        self.min_ranking_rank = config.get('min_ranking_rank', 0)  # 0=無制限

    def _min_proba(self, bet_type: str) -> float:
        """馬券種に対応する最低予測確率のしきい値を返す。

        Args:
            bet_type: 'win' / 'place' / 'umaren'

        Returns:
            float: 最低予測確率
        """
        return {
            'win': self.min_proba_win,
            'place': self.min_proba_place,
            'umaren': self.min_proba_umaren,
        }[bet_type]

    def _max_bets_per_race(self, bet_type: str) -> int:
        """馬券種に対応するレースあたり最大購入点数を返す。

        Args:
            bet_type: 'win' / 'place' / 'umaren'

        Returns:
            int: 最大購入点数（0 は無制限）
        """
        return {
            'win': self.max_bets_per_race_win,
            'place': self.max_bets_per_race_place,
            'umaren': self.max_bets_per_race_umaren,
        }[bet_type]

    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5,
        y_pred_binary: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        モデルの性能を評価

        Args:
            y_true: 正解ラベル
            y_pred_proba: 予測確率
            threshold: 閾値（y_pred_binary が指定された場合は無視される）
            y_pred_binary: EV基準など外部で計算済みの二値予測。指定した場合はthresholdより優先される。

        Returns:
            Dict: 評価指標の辞書
        """
        logger.info("モデル評価開始")

        # 二値予測を決定
        if y_pred_binary is not None:
            y_pred = y_pred_binary.astype(int)
        else:
            y_pred = (y_pred_proba >= threshold).astype(int)

        # 各種指標を計算。
        # AUC は単一クラス（正例のみ/負例のみ）で roc_auc_score が例外を投げるため
        # compute_auc に委譲し、その場合は nan を返す（学習終盤でのクラッシュを防ぐ）。
        brier = brier_score_loss(y_true, y_pred_proba)
        metrics = {
            'auc': self.compute_auc(y_true, y_pred_proba),
            'brier_score': brier,           # 角度4: 確率キャリブレーション精度（目標 < 0.08）
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold
        }

        # 混同行列
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm

        logger.info("評価指標:")
        logger.info(f"  AUC: {metrics['auc']:.4f}")
        logger.info(f"  Brier Score: {brier:.4f} (目標 < 0.08)")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1 Score: {metrics['f1']:.4f}")

        return metrics

    @staticmethod
    def compute_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """AUC を計算する。単一クラス（正例のみ/負例のみ）の場合は nan を返す。

        Args:
            y_true: 正解ラベル
            y_pred_proba: 予測確率

        Returns:
            float: AUC 値、または nan
        """
        true = y_true.values if isinstance(y_true, pd.Series) else y_true
        true = np.asarray(true)
        # 正例0（従来ガード）だけでなく負例0（全行正例）でも roc_auc_score は
        # 例外を投げるため、単一クラスをまとめて nan で弾く。
        pos = true.sum()
        if pos == 0 or pos == len(true):
            return float('nan')
        return float(roc_auc_score(true, y_pred_proba))

    def calculate_ranking_recovery_rate(
        self,
        test_eval_df: pd.DataFrame,
        temperature: float = 1.5,
        min_ev: float = 1.05,
    ) -> float:
        """ランキングスコアから softmax 確率を求め回収率を計算する。

        賭け判断には締切前オッズ（``odds_pre_win``）、的中時の払戻には確定払戻金
        （``payout_win``）を使い、リークを避ける。

        Args:
            test_eval_df: '_score', 'odds_pre_win', 'payout_win', 'finish_position',
                'race_id' 列を持つ DataFrame
            temperature: softmax の温度パラメータ
            min_ev: 購入する最低期待値

        Returns:
            float: 回収率（%）、計算不可の場合は 0.0
        """
        from src.predict.blend_ensemble import ranking_to_proba

        work = self._drop_rows_without_bet_odds(
            test_eval_df, np.zeros(len(test_eval_df)), 'win'
        ).drop(columns='_pred_proba_in')
        payout_odds_all = self._get_payout_odds_vec(work, 'win')
        work = work.assign(_payout_odds=payout_odds_all)

        total_bet = 0.0
        total_return = 0.0
        for _, race in work.groupby('race_id'):
            proba = ranking_to_proba(race['_score'].to_numpy(), temperature)
            odds = race['odds_pre_win'].to_numpy(dtype=float)
            hit_mask = proba * odds >= min_ev
            n = int(hit_mask.sum())
            if n == 0:
                continue
            # iterrows を廃止しレース内でベクトル集計
            total_bet += 100.0 * n
            finish = race['finish_position'].to_numpy()
            won = hit_mask & (finish == 1)
            total_return += 100.0 * race['_payout_odds'].to_numpy()[won].sum()
        return (total_return / total_bet * 100) if total_bet > 0 else 0.0

    def calculate_recovery_rate(
        self,
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        min_ev: float,
        bet_type: str = 'win'
    ) -> Dict:
        """
        回収率を計算（期待値ベース）

        Args:
            df: データフレーム（オッズ情報を含む）
            y_pred_proba: 予測確率
            min_ev: 購入する最低期待値（EV = 予測確率 × オッズ）
            bet_type: 賭けの種類（'win' or 'place'）

        Returns:
            Dict: 回収率と統計情報
        """
        if bet_type not in SUPPORTED_BET_TYPES:  # 検証を先頭へ（不正値で apply 内 raise しない）
            raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")

        logger.info(f"回収率計算開始 ({bet_type})")

        # 締切前オッズが欠損する行は評価対象外にする（odds_win での代用はしない）。
        # 代用すると賭け判断に確定オッズが混入しリークになるため。
        df = self._drop_rows_without_bet_odds(df, y_pred_proba, bet_type)
        y_pred_proba = df['_pred_proba_in'].to_numpy(dtype=float)
        df = df.drop(columns='_pred_proba_in')

        # 予測確率とEVを追加（apply を廃止しベクトル演算）
        df['pred_proba'] = y_pred_proba
        # 賭け判断は締切前オッズ、払戻は確定払戻金と、オッズ源を分離する
        df['odds_used'] = self._get_bet_odds_vec(df, bet_type)
        df['payout_odds'] = self._get_payout_odds_vec(df, bet_type)
        df['ev'] = df['pred_proba'] * df['odds_used']

        # bet_type に応じた確率フィルター閾値を選択
        min_proba = self._min_proba(bet_type)
        max_bets_per_race = self._max_bets_per_race(bet_type)

        # EV >= min_ev かつ pred_proba >= min_proba の馬にのみ賭ける
        bet_df = df[(df['ev'] >= min_ev) & (df['pred_proba'] >= min_proba)].copy()

        # ランキングモデルフィルター（ranking_rank が min_ranking_rank 以下の馬のみ）
        if self.min_ranking_rank > 0 and 'ranking_rank' in bet_df.columns:
            bet_df = bet_df[bet_df['ranking_rank'] <= self.min_ranking_rank]

        # レースあたり最大頭数フィルター（確率上位 max_bets_per_race 頭のみ残す）
        if max_bets_per_race > 0 and 'race_id' in bet_df.columns:
            bet_df = (
                bet_df
                .sort_values('pred_proba', ascending=False)
                .groupby('race_id', group_keys=False)
                .head(max_bets_per_race)
            )

        if len(bet_df) == 0:
            logger.warning("賭け対象の馬が0頭です")
            return {
                'recovery_rate': 0.0,
                'total_bet': 0,
                'total_return': 0,
                'profit': 0,
                'num_bets': 0,
                'num_wins': 0,
                'hit_rate': 0.0,
                'min_ev': min_ev
            }

        # 賭け金の計算（apply を廃止しベクトル演算）
        if self.use_kelly:
            bet_df['bet_amount'] = self._kelly_criterion_vec(
                bet_df['pred_proba'].to_numpy(dtype=float),
                bet_df['odds_used'].to_numpy(dtype=float),
                self.initial_bankroll,
            )
        else:
            bet_df['bet_amount'] = float(self.bet_unit)

        # 的中判定と払戻し計算（apply を廃止しベクトル演算）。
        # 払戻は確定払戻金ベースの payout_odds を使う（賭け判断の odds_used とは別系統）。
        bet_df['hit'] = self._is_hit_vec(bet_df, bet_type).astype(int)
        bet_df['return'] = bet_df['hit'] * bet_df['payout_odds'] * bet_df['bet_amount']

        # 統計計算
        total_bet = bet_df['bet_amount'].sum()
        total_return = bet_df['return'].sum()
        recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0.0

        num_bets = len(bet_df)
        num_wins = bet_df['hit'].sum()
        hit_rate = (num_wins / num_bets * 100) if num_bets > 0 else 0.0

        results = {
            'recovery_rate': recovery_rate,
            'total_bet': int(total_bet),
            'total_return': int(total_return),
            'profit': int(total_return - total_bet),
            'num_bets': num_bets,
            'num_wins': int(num_wins),
            'hit_rate': hit_rate,
            'min_ev': min_ev
        }

        logger.info("回収率統計:")
        logger.info(f"  回収率: {recovery_rate:.2f}%")
        logger.info(f"  総賭け金: ¥{total_bet:,}")
        logger.info(f"  総払戻金: ¥{total_return:,}")
        logger.info(f"  収支: ¥{results['profit']:,}")
        logger.info(f"  賭け回数: {num_bets} 回")
        logger.info(f"  的中数: {num_wins} 回")
        logger.info(f"  的中率: {hit_rate:.2f}%")

        return results

    def _drop_rows_without_bet_odds(
        self, df: pd.DataFrame, y_pred_proba: np.ndarray, bet_type: str
    ) -> pd.DataFrame:
        """締切前オッズが欠損する行を評価対象から除外する。

        CLAUDE.md の「補完で握りつぶさない」方針に従い、欠損を確定オッズで
        代用せず除外し、除外件数をログに残す。全行欠損なら停止する。

        Args:
            df: 評価対象 DataFrame
            y_pred_proba: 予測確率（df と同じ行数・行順）
            bet_type: 'win' / 'place' / 'umaren'

        Returns:
            pd.DataFrame: 有効行のみの DataFrame（予測確率を '_pred_proba_in' 列で保持）

        Raises:
            ValueError: 有効行が0件の場合
        """
        work = df.copy()
        work['_pred_proba_in'] = np.asarray(y_pred_proba, dtype=float)

        bet_col = BET_ODDS_COLS[bet_type]
        if bet_col not in work.columns:
            raise ValueError(
                f"賭け判断用の締切前オッズ列 '{bet_col}' がありません（bet_type={bet_type}）。"
                "src.models.odds_series.attach_meta_columns で結合してください。"
            )

        valid = work[bet_col].notna() & (work[bet_col] > 0)
        n_dropped = int((~valid).sum())
        if n_dropped > 0:
            logger.warning(
                f"[{bet_type}] 締切前オッズ欠損のため {n_dropped}/{len(work)} 行を"
                "回収率評価の対象外にしました"
            )
        if int(valid.sum()) == 0:
            raise ValueError(
                f"[{bet_type}] 締切前オッズが有効な行が0件です。"
                f"'{bet_col}' の結合と対象期間を確認してください。"
            )
        return work[valid].reset_index(drop=True)

    def _get_bet_odds_vec(self, df: pd.DataFrame, bet_type: str) -> np.ndarray:
        """賭け判断に使う締切前オッズ配列を返す（ベクトル版）。

        予測時点で入手可能な締切前オッズのみを使う。確定オッズ（odds_win）を
        賭け判断に使うとリークになるため使わない。

        Args:
            df: 締切前オッズ列を持つ DataFrame
            bet_type: 賭けタイプ（'win' / 'place' / 'umaren'）

        Returns:
            np.ndarray: 賭け判断用オッズ配列（float）

        Raises:
            ValueError: 賭けタイプが不正、または必要な列が無い場合
        """
        if bet_type not in SUPPORTED_BET_TYPES:
            raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")
        col = BET_ODDS_COLS[bet_type]
        if col not in df.columns:
            raise ValueError(
                f"賭け判断用の締切前オッズ列 '{col}' がありません（bet_type={bet_type}）"
            )
        return df[col].to_numpy(dtype=float)

    def _get_payout_odds_vec(self, df: pd.DataFrame, bet_type: str) -> np.ndarray:
        """的中時の払戻倍率（確定払戻金 / 100）を返す（ベクトル版）。

        払戻はレース結果なので、賭け判断には使わず的中後の払戻計算にのみ使う。

        Args:
            df: 確定払戻列を持つ DataFrame
            bet_type: 賭けタイプ（'win' / 'place' / 'umaren'）

        Returns:
            np.ndarray: 払戻倍率の配列（float）。非的中行は 0。

        Raises:
            ValueError: 賭けタイプが不正、または必要な列が無い場合
        """
        if bet_type not in SUPPORTED_BET_TYPES:
            raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")
        col = PAYOUT_COLS[bet_type]
        if col not in df.columns:
            raise ValueError(
                f"払戻列 '{col}' がありません（bet_type={bet_type}）。"
                "src.models.odds_series.attach_meta_columns で結合してください。"
            )
        # 払戻金は100円あたりなので 100 で割って倍率にする
        return df[col].fillna(0.0).to_numpy(dtype=float) / 100.0

    def _kelly_criterion_vec(
        self,
        p: np.ndarray,
        odds: np.ndarray,
        bankroll: float,
    ) -> np.ndarray:
        """ケリー基準の賭け金をベクトルで計算する（スカラー版と等価）。

        Args:
            p: 予測確率の配列
            odds: オッズの配列
            bankroll: 賭け金の基準となる資金

        Returns:
            np.ndarray: 推奨賭け金の配列
        """
        b = odds - 1.0
        with np.errstate(divide='ignore', invalid='ignore'):
            f = (p * b - (1.0 - p)) / b
        # odds <= 1.0 または p <= 0 は賭けない（スカラー版と同じ境界）
        f = np.where((odds <= 1.0) | (p <= 0.0), 0.0, f)
        f = np.maximum(f * self.kelly_fraction, 0.0)
        return np.minimum(f * bankroll, self.max_bet)

    def _is_hit_vec(self, df: pd.DataFrame, bet_type: str) -> np.ndarray:
        """的中判定をベクトルで行う。

        win / place は着順から、umaren はペア単位の ``target_umaren`` から判定する。

        Args:
            df: 'finish_position'（win/place）または 'target_umaren'（umaren）を持つ DataFrame
            bet_type: 賭けタイプ（'win' / 'place' / 'umaren'）

        Returns:
            np.ndarray: 的中なら True の bool 配列

        Raises:
            ValueError: 賭けタイプが不正、または必要な列が無い場合
        """
        if bet_type == 'umaren':
            # 馬連は1行1ペアなので着順ではなくペアの的中ラベルで判定する
            if 'target_umaren' not in df.columns:
                raise ValueError("馬連の的中判定に必要な 'target_umaren' 列がありません")
            return df['target_umaren'].to_numpy() == 1
        if bet_type not in ('win', 'place'):
            raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")
        if 'finish_position' not in df.columns:
            raise ValueError("的中判定に必要な 'finish_position' 列がありません")
        finish_position = df['finish_position'].to_numpy()
        if bet_type == 'win':
            return finish_position == 1
        # finish_position には出走取消・除外・失格を表す負値（-1/-2/-3）が入るため、
        # `<= 3` だけだとこれらを的中と誤判定する。下限も確認する。
        return (finish_position >= 1) & (finish_position <= 3)

    def _kelly_criterion(
        self,
        win_probability: float,
        odds: float,
        bankroll: float
    ) -> float:
        """
        ケリー基準で最適な賭け金を計算

        Args:
            win_probability: 勝率（予測確率）
            odds: オッズ
            bankroll: 現在の資金（賭け金の基準）

        Returns:
            float: 推奨賭け金
        """
        if odds <= 1.0 or win_probability <= 0:
            return 0.0

        # ケリー基準の計算
        # f* = (p * (odds - 1) - (1 - p)) / (odds - 1)
        kelly_fraction_calc = (
            (win_probability * (odds - 1) - (1 - win_probability)) / (odds - 1)
        )

        # ハーフケリーなどの調整
        kelly_fraction_calc *= self.kelly_fraction

        # 賭け金が負にならないように
        kelly_fraction_calc = max(0, kelly_fraction_calc)

        # bankrollに対するケリー割合で賭け金を計算し、max_betで上限を設定
        bet_amount = kelly_fraction_calc * bankroll
        bet_amount = min(bet_amount, self.max_bet)

        return bet_amount

    @staticmethod
    def _is_hit(finish_position: int, bet_type: str) -> bool:
        """
        的中判定を行う（スカラー版。simulate_betting の逐次処理から使用）。

        Args:
            finish_position: 着順
            bet_type: 賭けタイプ（'win' or 'place'）

        Returns:
            bool: 的中ならTrue
        """
        if bet_type == 'win':
            return finish_position == 1
        if bet_type == 'place':
            # finish_position には出走取消・除外・失格を表す負値（-1/-2/-3）が入るため、
            # `<= 3` だけだとこれらを的中と誤判定する。下限も確認する。
            return 1 <= finish_position <= 3
        raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")

    def optimize_min_ev(
        self,
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        bet_type: str = 'win',
        metric: str = 'recovery_rate'
    ) -> Tuple[float, Dict]:
        """
        回収率を最大化する最低期待値（min_ev）を探索

        EV = 予測確率 × オッズ。EV >= min_ev の馬のみ購入する。

        Args:
            df: データフレーム
            y_pred_proba: 予測確率
            bet_type: 賭けタイプ
            metric: 最適化する指標

        Returns:
            Tuple: (最適min_ev, 最適結果)
        """
        if bet_type not in SUPPORTED_BET_TYPES:
            raise ValueError(f"サポートされていない賭けタイプ: {bet_type}")

        logger.info("min_ev最適化開始")

        # min_ev に依存しない量（オッズ・EV・的中・賭け金）は1回だけ計算する。
        # 以前は候補40件それぞれで calculate_recovery_rate を全再計算していた。
        work = self._drop_rows_without_bet_odds(df, y_pred_proba, bet_type)
        work['pred_proba'] = work.pop('_pred_proba_in')
        # 賭け判断は締切前オッズ、払戻は確定払戻金
        work['odds_used'] = self._get_bet_odds_vec(work, bet_type)
        work['payout_odds'] = self._get_payout_odds_vec(work, bet_type)
        work['ev'] = work['pred_proba'] * work['odds_used']

        min_proba = self._min_proba(bet_type)
        max_bets_per_race = self._max_bets_per_race(bet_type)

        ev = work['ev'].to_numpy(dtype=float)
        proba = work['pred_proba'].to_numpy(dtype=float)
        odds = work['odds_used'].to_numpy(dtype=float)
        payout_odds = work['payout_odds'].to_numpy(dtype=float)
        hit = self._is_hit_vec(work, bet_type)
        if self.use_kelly:
            # ケリーの賭け金も締切前オッズで計算する（払戻オッズを使うとリーク）
            amount = self._kelly_criterion_vec(proba, odds, self.initial_bankroll)
        else:
            amount = np.full(len(work), float(self.bet_unit))
        ret = hit * payout_odds * amount

        # min_ev に依存しないフィルター（proba・ranking_rank）を事前マスク化
        base_mask = proba >= min_proba
        if self.min_ranking_rank > 0 and 'ranking_rank' in work.columns:
            base_mask &= work['ranking_rank'].to_numpy() <= self.min_ranking_rank

        # max_bets_per_race フィルターは母集団（EV通過後）に依存するため、
        # そのフィルターが有効な場合のみ calculate_recovery_rate にフォールバックする。
        use_fast_path = not (max_bets_per_race > 0 and 'race_id' in work.columns)

        ev_candidates = np.arange(1.05, 3.05, 0.05)
        best_min_ev = 1.05
        best_result: Optional[Dict] = None
        best_metric_value = -np.inf

        for min_ev in ev_candidates:
            if use_fast_path:
                mask = base_mask & (ev >= min_ev)
                n = int(mask.sum())
                if n == 0:
                    continue
                tb = float(amount[mask].sum())
                tr = float(ret[mask].sum())
                nw = int(hit[mask].sum())
                result = {
                    'recovery_rate': (tr / tb * 100) if tb > 0 else 0.0,
                    'total_bet': int(tb),
                    'total_return': int(tr),
                    'profit': int(tr - tb),
                    'num_bets': n,
                    'num_wins': nw,
                    'hit_rate': (nw / n * 100) if n > 0 else 0.0,
                    'min_ev': float(min_ev),
                }
            else:
                result = self.calculate_recovery_rate(df, y_pred_proba, min_ev, bet_type)
                if result['num_bets'] == 0:
                    continue

            metric_value = result.get(metric, 0)
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_min_ev = float(min_ev)
                best_result = result

        if best_result is None:
            # 全候補で賭け対象が0件。呼び出し側が best_result['...'] で
            # 即クラッシュしないよう、賭けなし相当の空結果を返す。
            logger.warning("全てのmin_ev候補で賭け対象が0件でした")
            best_result = self.calculate_recovery_rate(
                df, y_pred_proba, float(ev_candidates[-1]) + 1.0, bet_type
            )
            return best_min_ev, best_result

        logger.info(f"最適min_ev: {best_min_ev:.2f}")
        logger.info(f"最良{metric}: {best_metric_value:.2f}")

        return best_min_ev, best_result

    def simulate_betting(
        self,
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        min_ev: float,
        bet_type: str = 'win',
        initial_bankroll: float = 100000
    ) -> pd.DataFrame:
        """
        賭けのシミュレーション（時系列、期待値ベース）

        EV = 予測確率 × オッズ。EV >= min_ev の馬のみ購入する。

        Args:
            df: データフレーム
            y_pred_proba: 予測確率
            min_ev: 購入する最低期待値
            bet_type: 賭けタイプ
            initial_bankroll: 初期資金

        Returns:
            pd.DataFrame: シミュレーション結果
        """
        logger.info("賭けシミュレーション開始")

        df = self._drop_rows_without_bet_odds(df, y_pred_proba, bet_type)
        df['pred_proba'] = df.pop('_pred_proba_in')

        # 日付でソート
        df = df.sort_values('date').reset_index(drop=True)

        # bet_type に応じた確率フィルター閾値を選択
        min_proba = self._min_proba(bet_type)
        max_bets_per_race = self._max_bets_per_race(bet_type)

        # EV・確率フィルターを事前計算して _should_bet フラグを付与（apply を廃止）
        # 賭け判断は締切前オッズ、払戻は確定払戻金と分離する
        df['_odds_used'] = self._get_bet_odds_vec(df, bet_type)
        df['_payout_odds'] = self._get_payout_odds_vec(df, bet_type)
        df['_ev'] = df['pred_proba'] * df['_odds_used']
        df['_should_bet'] = (df['_ev'] >= min_ev) & (df['pred_proba'] >= min_proba)

        # ランキングモデルフィルター（ranking_rank が min_ranking_rank 以下の馬のみ）
        if self.min_ranking_rank > 0 and 'ranking_rank' in df.columns:
            df['_should_bet'] = df['_should_bet'] & (df['ranking_rank'] <= self.min_ranking_rank)

        # レースあたり最大頭数フィルター（確率上位 max_bets_per_race 頭のみ残す）
        # groupby().apply() を避け rank ベースで計算（カラム消失を防ぐため）
        if max_bets_per_race > 0 and 'race_id' in df.columns:
            # _should_bet な馬のみ対象にレース内確率ランクを付与（対象外は NaN）
            proba_rank = (
                df.where(df['_should_bet'])
                .groupby('race_id')['pred_proba']
                .rank(ascending=False, method='first')
            )
            df['_should_bet'] = df['_should_bet'] & (proba_rank <= max_bets_per_race)

        # シミュレーション結果を記録。
        # bankroll が逐次依存のため逐次処理は必須だが、賭け対象のみに絞り
        # iterrows より高速な itertuples で走査する（全行 iterrows を廃止）。
        results = []
        current_bankroll = initial_bankroll

        has_horse_name = 'horse_name' in df.columns
        # itertuples は先頭アンダースコアの列名を _N にリネームするため、
        # 走査に使う列だけ有効な識別子へリネームしてから抽出する。
        cols = [
            'date', 'race_id', 'pred_proba', 'finish_position',
            '_odds_used', '_payout_odds', '_ev',
        ]
        if has_horse_name:
            cols.append('horse_name')
        bet_rows = df.loc[df['_should_bet'], cols].rename(
            columns={'_odds_used': 'odds_used', '_payout_odds': 'payout_odds', '_ev': 'ev'}
        )
        for row in bet_rows.itertuples(index=False):
            odds = row.odds_used
            ev = row.ev

            # 賭け金計算（変動ケリー：現在資金を基準に算出）
            if self.use_kelly:
                bet_amount = self._kelly_criterion(row.pred_proba, odds, current_bankroll)
            else:
                bet_amount = self.bet_unit

            # 資金不足チェック
            if bet_amount > current_bankroll:
                bet_amount = current_bankroll

            if bet_amount <= 0:
                continue

            # 的中判定と払戻し計算（払戻は確定払戻金ベースの倍率を使う）
            hit = self._is_hit(row.finish_position, bet_type)
            return_amount = bet_amount * row.payout_odds if hit else 0

            # 資金更新
            current_bankroll = current_bankroll - bet_amount + return_amount

            # 記録
            results.append({
                'date': row.date,
                'race_id': row.race_id,
                'horse_name': getattr(row, 'horse_name', '') if has_horse_name else '',
                'pred_proba': row.pred_proba,
                'ev': ev,
                'bet_amount': bet_amount,
                'hit': hit,
                'return': return_amount,
                'bankroll': current_bankroll
            })

        sim_df = pd.DataFrame(results)

        if not sim_df.empty:
            final_bankroll = sim_df['bankroll'].iloc[-1]
            total_return_rate = (final_bankroll / initial_bankroll - 1) * 100

            logger.info(f"初期資金: ¥{initial_bankroll:,}")
            logger.info(f"最終資金: ¥{final_bankroll:,.0f}")
            logger.info(f"総収益率: {total_return_rate:.2f}%")

        return sim_df

    # 馬券種ごとのデフォルト平均オッズ（市場平均倍率）
    _DEFAULT_COMBO_ODDS: Dict[str, float] = {
        'wide': 10.0,              # ワイド: 市場平均 850〜1,160円 ≈ 10倍
        'wide_ranking': 10.0,      # ワイド(ランキングtop-2): 同上
        'umaren': 24.0,            # 馬連: 市場平均 1,910〜2,870円 ≈ 24倍
        'sanrenfuku': 70.0,        # 三連複(1頭軸流し): 市場平均 5,400〜8,860円 ≈ 70倍
        'sanrenfuku_top3': 70.0,   # 三連複(top-3ボックス): 同上
    }

    def calculate_combo_recovery_rate(
        self,
        df: pd.DataFrame,
        y_pred_proba: np.ndarray,
        bet_type: str = 'wide',
        num_opponents: int = 5,
        avg_odds: Optional[float] = None,
        ranking_col: str = 'ranking_rank',
    ) -> Dict:
        """
        組み合わせ馬券（ワイド・馬連・三連複）のバックテストを行い回収率を計算する。

        実際のオッズが利用できない場合は市場平均オッズを使って推定する。

        ワイド:
            同一レース内で pred_proba 上位2頭の組み合わせを購入。
            両馬が3着以内に入れば的中。
        馬連1頭軸流し:
            ranking_col == 1 の馬を軸に、pred_proba 上位 num_opponents 頭へ流す。
            軸馬と相手馬のいずれかがともに2着以内に入れば的中。
        三連複1頭軸流し:
            ranking_col == 1 の馬を軸に、pred_proba 上位 num_opponents 頭
            の全ペア（C(n,2) 点）を購入。
            軸馬と当該ペア2頭が全て3着以内に入れば的中。

        Args:
            df: レースデータ（finish_position・race_id 列を含む）
            y_pred_proba: 各行の予測確率（df と同順）
            bet_type: 'wide' / 'umaren' / 'sanrenfuku'
            num_opponents: 軸流しで選ぶ相手馬の最大頭数
            avg_odds: 使用する平均オッズ（None の場合は馬券種ごとのデフォルト値）
            ranking_col: ランキング順位列名（'ranking_rank' など）

        Returns:
            Dict: recovery_rate / hit_rate / num_bets / num_wins / total_bet /
                  total_return / profit / bet_type を含む辞書
        """
        if bet_type not in ('wide', 'wide_ranking', 'umaren', 'sanrenfuku', 'sanrenfuku_top3'):
            raise ValueError(f"サポートされていない馬券種: {bet_type}")

        logger.info(f"組み合わせ馬券バックテスト開始 ({bet_type})")

        df = df.copy()
        df['pred_proba'] = y_pred_proba

        odds = avg_odds if avg_odds is not None else self._DEFAULT_COMBO_ODDS[bet_type]
        bet_amount_per_ticket = self.bet_unit

        total_tickets = 0
        total_hits = 0
        total_bet = 0
        total_return = 0.0

        for _race_id, race in df.groupby('race_id'):
            race = race.sort_values('pred_proba', ascending=False).reset_index(drop=True)

            if bet_type == 'wide':
                tickets, hits = self._eval_wide(race)
            elif bet_type == 'wide_ranking':
                tickets, hits = self._eval_wide_ranking(race, ranking_col)
            elif bet_type == 'umaren':
                tickets, hits = self._eval_umaren(race, num_opponents, ranking_col)
            elif bet_type == 'sanrenfuku_top3':
                tickets, hits = self._eval_sanrenfuku_top3(race, ranking_col)
            else:  # sanrenfuku
                tickets, hits = self._eval_sanrenfuku(race, num_opponents, ranking_col)

            if tickets == 0:
                continue

            race_bet = tickets * bet_amount_per_ticket
            race_return = hits * odds * bet_amount_per_ticket

            total_tickets += tickets
            total_hits += hits
            total_bet += race_bet
            total_return += race_return

        recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0.0
        hit_rate = (total_hits / total_tickets * 100) if total_tickets > 0 else 0.0

        results = {
            'bet_type': bet_type,
            'recovery_rate': round(recovery_rate, 2),
            'hit_rate': round(hit_rate, 2),
            'num_bets': total_tickets,
            'num_wins': total_hits,
            'total_bet': int(total_bet),
            'total_return': int(total_return),
            'profit': int(total_return - total_bet),
            'avg_odds_used': odds,
            'num_opponents': num_opponents,
        }

        logger.info(f"  回収率(推定): {recovery_rate:.2f}%")
        logger.info(f"  的中率: {hit_rate:.2f}%")
        logger.info(f"  総点数: {total_tickets}")
        logger.info(f"  的中数: {total_hits}")
        logger.info(f"  使用オッズ(平均): {odds:.1f}倍")

        return results

    @staticmethod
    def _eval_wide(race: pd.DataFrame) -> Tuple[int, int]:
        """ワイドの的中判定（pred_proba 上位2頭の組み合わせ）。

        Args:
            race: 1レース分のデータ（pred_proba 降順ソート済み）

        Returns:
            Tuple[int, int]: (購入点数, 的中数)
        """
        candidates = race
        if len(candidates) < 2:
            return 0, 0

        top2 = candidates.head(2)
        tickets = 1  # 1ペアを購入
        # finish_position の負値（取消・除外・失格）を的中扱いしないよう下限も確認する。
        hit = int(top2['finish_position'].between(1, 3).all())
        return tickets, hit

    @staticmethod
    def _eval_wide_ranking(race: pd.DataFrame, ranking_col: str) -> Tuple[int, int]:
        """ワイド（ランキング基準）の的中判定: ranking_col top-2の組み合わせを購入。

        角度6: ランキングtop-2をワイドで購入する戦略。
        両馬が3着以内に入れば的中（1点¥100）。

        Args:
            race: 1レース分のデータ
            ranking_col: ランキング順位列名

        Returns:
            Tuple[int, int]: (購入点数, 的中数)
        """
        if ranking_col not in race.columns:
            return 0, 0
        top2 = race.nsmallest(2, ranking_col)
        if len(top2) < 2:
            return 0, 0
        tickets = 1
        # finish_position の負値（取消・除外・失格）を的中扱いしないよう下限も確認する。
        hit = int(top2['finish_position'].between(1, 3).all())
        return tickets, hit

    @staticmethod
    def _eval_sanrenfuku_top3(race: pd.DataFrame, ranking_col: str) -> Tuple[int, int]:
        """三連複top-3ボックスの的中判定: ranking_col top-3の3頭全員が3着以内なら的中。

        角度6: ランキングtop-3の全組み合わせは1通り（¥100投資）。
        3頭全員が3着以内に入れば的中。

        Args:
            race: 1レース分のデータ
            ranking_col: ランキング順位列名

        Returns:
            Tuple[int, int]: (購入点数, 的中数)
        """
        if ranking_col not in race.columns:
            return 0, 0
        top3 = race.nsmallest(3, ranking_col)
        if len(top3) < 3:
            return 0, 0
        tickets = 1
        # finish_position の負値（取消・除外・失格）を的中扱いしないよう下限も確認する。
        hit = int(top3['finish_position'].between(1, 3).all())
        return tickets, hit

    @staticmethod
    def _eval_umaren(
        race: pd.DataFrame,
        num_opponents: int,
        ranking_col: str,
    ) -> Tuple[int, int]:
        """馬連1頭軸流しの的中判定。

        軸馬: ranking_col が最小（1位）の馬。
        相手: pred_proba 上位 num_opponents 頭（軸除く）。
        的中: 軸馬と相手馬のいずれかが共に2着以内。

        Args:
            race: 1レース分のデータ
            num_opponents: 相手馬の最大頭数
            ranking_col: ランキング順位列名

        Returns:
            Tuple[int, int]: (購入点数, 的中数)
        """
        if ranking_col not in race.columns:
            return 0, 0

        axis_mask = race[ranking_col] == race[ranking_col].min()
        if not axis_mask.any():
            return 0, 0

        axis = race[axis_mask].iloc[0]
        opponents = race[~axis_mask].head(num_opponents)

        if opponents.empty:
            return 0, 0

        tickets = len(opponents)
        axis_in_top2 = axis['finish_position'] <= 2
        hits = 0
        if axis_in_top2:
            hits = int((opponents['finish_position'] <= 2).any())
        return tickets, hits

    @staticmethod
    def _eval_sanrenfuku(
        race: pd.DataFrame,
        num_opponents: int,
        ranking_col: str,
    ) -> Tuple[int, int]:
        """三連複1頭軸流しの的中判定。

        軸馬: ranking_col が最小（1位）の馬。
        相手: pred_proba 上位 num_opponents 頭（軸除く）の全ペア。
        的中: 軸馬と当該ペア2頭が全て3着以内に入った場合。

        Args:
            race: 1レース分のデータ（pred_proba 降順ソート済み）
            num_opponents: 相手馬の最大頭数
            ranking_col: ランキング順位列名

        Returns:
            Tuple[int, int]: (購入点数, 的中数)
        """
        if ranking_col not in race.columns:
            return 0, 0

        axis_mask = race[ranking_col] == race[ranking_col].min()
        if not axis_mask.any():
            return 0, 0

        axis = race[axis_mask].iloc[0]
        opponents = race[~axis_mask].head(num_opponents)

        n = len(opponents)
        if n < 2:
            return 0, 0

        tickets = n * (n - 1) // 2  # C(n, 2)

        # 的中ペア数は itertools ループ不要。相手のうち3着以内が k 頭なら C(k,2)。
        # finish_position の負値（取消・除外・失格）を的中扱いしないよう下限も確認する。
        hits = 0
        if 1 <= axis['finish_position'] <= 3:
            opp_pos = opponents['finish_position'].to_numpy()
            k = int(((opp_pos >= 1) & (opp_pos <= 3)).sum())
            hits = k * (k - 1) // 2

        return tickets, hits

    def print_summary(self, metrics: Dict, recovery_results: Dict):
        """
        評価結果のサマリーを表示

        Args:
            metrics: モデル評価指標
            recovery_results: 回収率結果
        """
        print("\n" + "="*50)
        print("モデル評価サマリー")
        print("="*50)
        print("\n【予測性能】")
        print(f"  AUC-ROC:    {metrics['auc']:.4f}")
        brier = metrics.get('brier_score')
        if brier is not None:
            flag = " [OK]" if brier < 0.08 else " [NG] 目標0.08未達"
            print(f"  Brier Score: {brier:.4f}{flag}")
        print(f"  Accuracy:   {metrics['accuracy']:.4f}")
        print(f"  Precision:  {metrics['precision']:.4f}")
        print(f"  Recall:     {metrics['recall']:.4f}")
        print(f"  F1 Score:   {metrics['f1']:.4f}")

        print("\n【回収率】")
        print(f"  回収率:     {recovery_results['recovery_rate']:.2f}%")
        print(f"  的中率:     {recovery_results['hit_rate']:.2f}%")
        print(f"  総賭け金:   {recovery_results['total_bet']:,}円")
        print(f"  総払戻金:   {recovery_results['total_return']:,}円")
        print(f"  収支:       {recovery_results['profit']:,}円")
        print(f"  賭け回数:   {recovery_results['num_bets']} 回")
        print(f"  的中数:     {recovery_results['num_wins']} 回")
        if 'min_ev' in recovery_results:
            print(f"  最低EV:     {recovery_results['min_ev']:.2f}")
        print("="*50 + "\n")

    def calculate_value_bet_recovery_rate(
        self,
        df: pd.DataFrame,
        ranking_scores: np.ndarray,
        min_rank_disagreement: int = 2,
        max_model_rank: int = 3,
        bet_type: str = 'win',
    ) -> Dict:
        """
        バリューベット回収率を計算する。

        ランキングモデルが市場（オッズ）よりも高く評価している馬（rank_vs_odds_disagreement
        が大きい馬）に絞ってバックテストを行う。
        「モデルが自信を持っているのに市場では過小評価されている馬」が
        バリューベットの対象候補となる。

        Args:
            df: レースデータ（race_id, finish_position, odds_win 列が必須）
            ranking_scores: ランキングモデルのスコア（df と同順）
            min_rank_disagreement: モデルランクがオッズランクより何位以上良ければ対象とするか
            max_model_rank: モデルランクが何位以内の馬を対象とするか
            bet_type: 賭けタイプ（'win' or 'place'）

        Returns:
            Dict: recovery_rate / hit_rate / num_bets / num_wins /
                  total_bet / total_return / profit を含む辞書
        """
        logger.info(
            f"バリューベット回収率計算開始 "
            f"(min_rank_disagreement={min_rank_disagreement}, "
            f"max_model_rank={max_model_rank}, bet_type={bet_type})"
        )

        if 'odds_win' not in df.columns:
            logger.warning("odds_win カラムがないためバリューベット計算をスキップします")
            return {
                'recovery_rate': 0.0, 'total_bet': 0, 'total_return': 0,
                'num_bets': 0, 'num_wins': 0, 'hit_rate': 0.0,
                'profit': 0,
            }

        df = df.copy()
        df['_ranking_score'] = ranking_scores

        # per-race copy + apply を廃止し、groupby().rank() で一括計算する。
        grp = df.groupby('race_id')
        model_rank = grp['_ranking_score'].rank(ascending=False, method='min')
        odds_rank = grp['odds_win'].rank(ascending=True, method='min')
        # 乖離度: 正値 = モデルが市場より高評価 → バリューベット候補
        disagreement = odds_rank - model_rank

        bet_mask = (model_rank <= max_model_rank) & (disagreement >= min_rank_disagreement)
        bet_df = df[bet_mask].copy()

        if bet_df.empty:
            logger.warning("バリューベット対象馬が0頭です")
            return {
                'recovery_rate': 0.0, 'total_bet': 0, 'total_return': 0,
                'num_bets': 0, 'num_wins': 0, 'hit_rate': 0.0,
                'profit': 0,
            }

        # odds/的中はベクトル演算。バリューベットは EV 不明のため固定賭け金。
        # 払戻は確定払戻金ベース（odds_win は上の市場ランク算出にのみ使う）。
        bet_df['_payout_odds'] = self._get_payout_odds_vec(bet_df, bet_type)
        bet_df['_hit'] = self._is_hit_vec(bet_df, bet_type).astype(int)
        bet_df['_bet_amount'] = float(self.bet_unit)
        bet_df['_return'] = bet_df['_hit'] * bet_df['_payout_odds'] * bet_df['_bet_amount']

        total_bet = float(bet_df['_bet_amount'].sum())
        total_return = float(bet_df['_return'].sum())
        recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0.0
        num_bets = len(bet_df)
        num_wins = int(bet_df['_hit'].sum())
        hit_rate = (num_wins / num_bets * 100) if num_bets > 0 else 0.0

        results = {
            'recovery_rate': round(recovery_rate, 2),
            'hit_rate': round(hit_rate, 2),
            'num_bets': num_bets,
            'num_wins': num_wins,
            'total_bet': int(total_bet),
            'total_return': int(total_return),
            'profit': int(total_return - total_bet),
            'min_rank_disagreement': min_rank_disagreement,
            'max_model_rank': max_model_rank,
            'bet_type': bet_type,
        }

        logger.info(f"バリューベット回収率: {recovery_rate:.2f}%")
        logger.info(f"  的中率: {hit_rate:.2f}%, 賭け数: {num_bets}, 的中数: {num_wins}")

        return results
