"""単勝モデル戦略の購入判定ルールと賭け金計算。

検証内容は ``docs/report/anaba_strategy_report/anaba_strategy_report.md``。

購入条件（利用者の指示による採用値）::

    [1] EV = 予測確率 × 判断時点オッズ >= 1.05
    [2] 判断時点オッズが 10倍以上 150倍以下
    [3] log(判断時点オッズ / 5mオッズ) <= -0.05   ← オッズが約5%以上下がった
    [4] レースの判断時点の総投票額が 5億円以上
    [5] 条件を満たす馬が複数いるレースでは、期待値が最も高い1頭のみ

    賭け金は Flat ベット（1点 100円）

IMPORTANT (時点の一貫性):
    選別に使う情報は判断時点（当日は 2m）のスナップショットまでで、
    EV に使うオッズも同じ時点を使う。確定オッズ・着順・払戻は選別に
    一切使わない。

NOTE (レポートの検証条件との差分):
    レポートの検証は EV>=1.20・1レース最大3点・fractional Kelly で
    回収率 136.98%（95%CI [109.3, 161.4]）だった。本実装は利用者の指示により
    EV>=1.05・1レース1点・Flat 100円とし、さらにレポート §5.1 の知見
    （厚いプールほど回収できる）に基づく総投票額5億円以上のフィルタを
    加えている。**この組み合わせ自体は未検証**であり、レポート §9-7 が
    「§5 のレース条件は購入ルールに入れておらず、組み込むには新しい
    データでの追検証が要る」と明記している点に注意すること。
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.simulator.features import DRIFT_COL, ODDS_COL, POOL_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 採用しきい値（利用者の指示による）
# ---------------------------------------------------------------------------

# [1] 期待値の下限（予測確率 × 判断時点オッズ）
DEFAULT_MIN_EV: float = 1.05

# [2] 判断時点オッズの下限。これ未満は妙味が小さい（レポート §1.2）
DEFAULT_ODDS_MIN: float = 10.0

# [2] 判断時点オッズの上限。これ超は的中率が落ちて期待値が崩れる
# （レポート §1.1: 150倍超の回収率は 41.8%）
DEFAULT_ODDS_MAX: float = 150.0

# [3] ドリフトの上限。負値なのでオッズが下がった（資金が入った）馬だけが残る
DEFAULT_MAX_DRIFT: float = -0.05

# [4] レースの判断時点の総投票額の下限（円）。レポート §5.1 の知見
#
# IMPORTANT (当日データでは絶対額が使えない):
#     学習データ（JV-Link 履歴 0B41）と当日データ（速報 0B31）で
#     ``hyosu_total`` のスケールが 9〜13倍違う（競馬場を揃えても同じ）。
#     5億円を当日データに当てると1レースも通らない。
#     当日運用では :data:`DEFAULT_MIN_POOL_PCTILE` を使うこと。
#     詳細は docs/report/pool_by_race_number_report.md §6。
DEFAULT_MIN_POOL_YEN: float = 5e8

# [4-代替] 総投票額の「その日・その競馬場の中での分位点」の下限（0〜1）。
# 絶対額ではなく相対順位なので、データ区分によるスケール差の影響を受けない。
# None なら絶対額（min_pool_yen）で判定する。
#
# NOTE (0.40 の根拠):
#     学習データの1m時点で5億円は上位59.5%に相当する
#     （= 分位点 0.405 以上）。同じ割合を残す設定である。
#     ただしこの対応づけ自体は未検証で、1開催日22レースの当日データしか
#     手元に無いため精度は保証できない。
DEFAULT_MIN_POOL_PCTILE: float | None = None

# [5] 1レースあたりの購入点数
DEFAULT_MAX_BETS_PER_RACE: int = 1

# Flat ベットの1点あたり金額（円）
DEFAULT_STAKE_YEN: float = 100.0

# ---------------------------------------------------------------------------
# 見送り理由のラベル（全馬 CSV の ``skip_reason`` に入る）
# ---------------------------------------------------------------------------

REASON_BET = ''                         # 購入する
REASON_POOL = 'pool_below_min'          # レース総投票額が下限未満
REASON_EV = 'ev_below_threshold'        # EV が下限未満
REASON_ODDS_LOW = 'odds_below_min'      # オッズが下限未満
REASON_ODDS_HIGH = 'odds_above_max'     # オッズが上限超
REASON_DRIFT = 'odds_drop_below_5pct'   # 5mからのオッズ下落が約5%未満（max_drift 未達）
REASON_MAX_BETS = 'not_top_ev_in_race'  # レース内で期待値1位でない

# 理由が複数あるときの区切り（例: ``pool_below_min|ev_below_threshold``）
REASON_SEP = '|'


@dataclass(frozen=True)
class WinRule:
    """単勝選別ルール。

    Attributes:
        min_ev: 期待値の下限（``予測確率 × 判断時点オッズ``）
        odds_min: 判断時点オッズの下限
        odds_max: 判断時点オッズの上限
        max_drift: ``log(判断時点オッズ / 5mオッズ)`` の上限。
            負値なのでオッズが下がった（資金が入った）馬だけが残る
        min_pool_yen: レースの判断時点の総投票額の下限（円）。
            ``min_pool_pctile`` が設定されている場合は使わない
        min_pool_pctile: 総投票額の「その日の中での分位点」の下限（0〜1）。
            設定すると絶対額ではなくこちらで判定する。データ区分による
            スケール差の影響を受けないため、当日運用で使う
        max_bets_per_race: 1レースあたりの最大点数（期待値上位から採用）
        stake_yen: Flat ベットの1点あたり金額（円）
    """

    min_ev: float = DEFAULT_MIN_EV
    odds_min: float = DEFAULT_ODDS_MIN
    odds_max: float = DEFAULT_ODDS_MAX
    max_drift: float = DEFAULT_MAX_DRIFT
    min_pool_yen: float = DEFAULT_MIN_POOL_YEN
    min_pool_pctile: float | None = DEFAULT_MIN_POOL_PCTILE
    max_bets_per_race: int = DEFAULT_MAX_BETS_PER_RACE
    stake_yen: float = DEFAULT_STAKE_YEN

    def describe(self) -> str:
        """ログ・CSV のヘッダに出す1行の要約を返す。"""
        return (
            f'EV>={self.min_ev:.2f} / '
            f'オッズ {self.odds_min:g}〜{self.odds_max:g}倍 / '
            f'ドリフト<={self.max_drift:+.2f} / '
            + (f'総投票額が日内上位{(1 - self.min_pool_pctile) * 100:.0f}% / '
               if self.min_pool_pctile is not None
               else f'総投票額>={self.min_pool_yen / 1e8:.1f}億円 / ') +
            f'1レース{self.max_bets_per_race}点 / '
            f'Flat {self.stake_yen:,.0f}円'
        )


def _join_reasons(flags: Dict[str, pd.Series], index: pd.Index) -> pd.Series:
    """立っている見送り理由を ``flags`` の順に連結する。

    理由を上書きせず全て残すことで、「プールと EV の両方で落ちた」のような
    複合的な見送りを CSV から読み取れるようにする。

    Args:
        flags: 理由ラベル -> その理由で見送るかの bool Series
        index: 戻り値の index

    Returns:
        pd.Series: ``REASON_SEP`` 区切りの理由（購入候補は ``REASON_BET``）
    """
    joined = pd.Series(REASON_BET, index=index, dtype='object')
    for label, failed in flags.items():
        hit = failed.reindex(index, fill_value=False).fillna(False).to_numpy()
        if not hit.any():
            continue
        prefix = joined[hit]
        joined[hit] = np.where(
            prefix == REASON_BET, label, prefix + REASON_SEP + label
        )
    return joined


def apply_rule(
    df: pd.DataFrame, proba: np.ndarray, rule: WinRule | None = None
) -> pd.DataFrame:
    """全馬に ``ev`` / ``skip_reason`` を付けて返す。

    購入しない馬も落とさず理由を付けて返す。当日の意思決定では
    「なぜ買わなかったか」が確認できる必要があるため。

    Args:
        df: ``race_id`` / ``ODDS_COL`` / ``DRIFT_COL`` / ``POOL_COL`` を持つ表
        proba: 校正済み予測確率（``df`` と同じ行順）
        rule: 選別ルール。省略時は採用値

    Returns:
        pd.DataFrame: ``pred_proba`` / ``odds_used`` / ``ev`` /
            ``skip_reason`` を付けた表（レース・確率順）

    Raises:
        ValueError: 必須列が欠けている、または判断に使う値に欠損がある場合
    """
    rule = rule or WinRule()
    need = ['race_id', ODDS_COL, DRIFT_COL, POOL_COL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(
            f'選別に必要な列がありません: {missing}。補完せず処理を停止します。'
        )

    work = df.copy().reset_index(drop=True)
    work['pred_proba'] = np.asarray(proba, dtype='float64')
    # 賭け判断オッズは特徴量と同じ判断時点（時点のズレを利用しない）
    work['odds_used'] = work[ODDS_COL].astype('float64')
    work['ev'] = work['pred_proba'] * work['odds_used']

    n_bad = int(
        work[['pred_proba', 'odds_used', DRIFT_COL, POOL_COL]]
        .isna().any(axis=1).sum()
    )
    if n_bad:
        raise ValueError(
            f'予測確率・オッズ・ドリフト・総投票額に欠損がある行が {n_bad} 件'
            ' あります。補完せず処理を停止します。'
        )

    # --- [4] 総投票額の判定。絶対額か日内分位点のどちらかを使う ---
    if rule.min_pool_pctile is not None:
        _warn_if_too_few_races_for_pctile(work, rule)
        # レース単位の分位点を取る（馬単位のままだと頭数の多いレースが重く数えられる）
        pool_by_race = work.groupby('race_id')[POOL_COL].first()
        # rank(pct=True) は 1/n 〜 1.0 を返し 0 を取らない。``>`` で比較すると
        # 「上位 (1 - pctile) 割」がちょうど残る（``>=`` だと1レース多く残る）。
        rank = pool_by_race.rank(pct=True)
        work['pool_pctile'] = work['race_id'].map(rank)
        pool_failed = work['pool_pctile'] <= rule.min_pool_pctile
    else:
        work['pool_pctile'] = np.nan
        pool_failed = work[POOL_COL] < rule.min_pool_yen

    # --- [1]〜[4] の条件判定。落ちた理由は全て残す ---
    work['skip_reason'] = _join_reasons(
        {
            REASON_POOL: pool_failed,
            REASON_ODDS_LOW: work['odds_used'] < rule.odds_min,
            REASON_ODDS_HIGH: work['odds_used'] > rule.odds_max,
            REASON_DRIFT: work[DRIFT_COL] > rule.max_drift,
            REASON_EV: work['ev'] < rule.min_ev,
        },
        work.index,
    )

    # --- [5] レース内で期待値上位のみ採用 ---
    candidate = work['skip_reason'] == REASON_BET
    if candidate.any():
        ranks = (
            work.loc[candidate]
            .groupby('race_id')['ev']
            .rank(ascending=False, method='first')
        )
        dropped = ranks[ranks > rule.max_bets_per_race].index
        work.loc[dropped, 'skip_reason'] = REASON_MAX_BETS

    _log_funnel(work, rule)
    _warn_if_pool_filter_blocks_everything(work, rule)
    return work.sort_values(
        ['race_id', 'ev'], ascending=[True, False]
    ).reset_index(drop=True)


# 分位点モードが意味を持つのに必要な最小レース数。
# 1レースだけだと順位は必ず1.0になり、フィルタが何もしなくなる。
MIN_RACES_FOR_PCTILE: int = 6


def _warn_if_too_few_races_for_pctile(
    work: pd.DataFrame, rule: WinRule
) -> None:
    """レース数が少なすぎて分位点フィルタが無意味になる場合に警告する。

    IMPORTANT (レース単位で予測する経路では分位点が効かない):
        :mod:`src.pipeline.daily` は 2m スナップショットが保存されたレースを
        1〜数レースずつ判断する。その状態で日内分位点を取ると、
        **渡されたレースの中での順位**になってしまう。極端な例として
        1レースだけ渡されると順位は必ず 1.0 になり、
        **フィルタが何も弾かなくなる**（＝総投票額の条件が消える）。

        これは黙って起きると「条件を満たしたから買った」と誤解するため、
        ここで明示的に警告する。当日のレース単位運用で分位点を使うなら、
        ``min_pool_pctile`` を null にして絶対額
        （当日データで測り直した値）を使うほうが安全である。
        詳細は docs/report/pool_by_race_number_report.md §6.2 の NOTE。
    """
    n_races = int(work['race_id'].nunique())
    if n_races >= MIN_RACES_FOR_PCTILE:
        return
    logger.error(
        '総投票額フィルタを日内分位点（上位%.0f%%）で判定しようとしましたが、'
        ' 対象が %d レースしかありません（最低 %d レース必要）。'
        ' この状態では順位がほぼ全レース上位になり、'
        ' **総投票額の条件が実質無効になります**。'
        ' レース単位で判断する運用では config.yaml の'
        ' strategy.min_pool_pctile を null にし、当日データで検証した'
        ' strategy.min_pool_yen を使ってください。',
        (1 - rule.min_pool_pctile) * 100, n_races, MIN_RACES_FOR_PCTILE,
    )


def _warn_if_pool_filter_blocks_everything(
    work: pd.DataFrame, rule: WinRule
) -> None:
    """総投票額フィルタが全レースを落としている場合に強く警告する。

    IMPORTANT (学習データと当日データで総票数のスケールが違う):
        実測では、当日の JV-Link 速報（``0B31``）が返す ``hyosu_total`` は
        学習データ（履歴 ``0B41``）の同等レースより **5〜12倍小さい**。

        ====================  ==============  ==============
        時点                  学習(履歴)中央値  当日(速報)中央値
        ====================  ==============  ==============
        5m                    2.15億円        0.29億円
        1m                    5.73億円        0.46億円
        2m                    （データ無し）   0.40億円
        ====================  ==============  ==============

        オッズ自体はどちらも控除率21%前後で整合しており、壊れていない。
        総票数の**絶対値だけ**がスケール違いである（当日の速報値は
        累計のレース総投票額ではなく、部分的な値を返していると見られる）。

        この結果、5億円という条件を当日データへそのまま当てると
        **1レースも通らない**（実測の当日最大値は1.75億円）。
        黙って0点を返し続けると「購入機会が無かった」と誤解されるため、
        ここで明示的に警告する。

        恒久的な対処は次のいずれかである（どちらも本実装の範囲外）。
        1. 当日取得を履歴と同じ ``0B41`` に揃え、総票数の定義を一致させる
        2. 当日データで検証し直した別のしきい値を ``config.yaml`` の
           ``strategy.min_pool_yen`` に設定する
    """
    if rule.min_pool_pctile is not None:
        return                      # 分位点モードは必ず一定割合が通るので対象外
    pool_by_race = work.groupby('race_id')[POOL_COL].first()
    if not len(pool_by_race) or (pool_by_race >= rule.min_pool_yen).any():
        return
    logger.error(
        '総投票額フィルタ（>=%s円）を通るレースが1つもありません。'
        ' 当日の最大値は %s円（%.2f億円）です。'
        ' 学習データ（履歴0B41）と当日データ（速報0B31）で総票数の'
        ' スケールが違う既知の問題の可能性があります。'
        ' config.yaml の strategy.min_pool_yen を当日データで検証した値に'
        ' 変更するか、当日取得を履歴と同じデータ区分に揃えてください。',
        f'{rule.min_pool_yen:,.0f}', f'{pool_by_race.max():,.0f}',
        pool_by_race.max() / 1e8,
    )


def _log_funnel(work: pd.DataFrame, rule: WinRule) -> None:
    """どのフィルタで何頭落ちたかをログに出す（当日の確認用）。"""
    n = len(work)
    counts = {
        label: int(
            work['skip_reason'].str.split(REASON_SEP).apply(lambda r: label in r).sum()
        )
        for label in (
            REASON_POOL, REASON_ODDS_LOW, REASON_ODDS_HIGH,
            REASON_DRIFT, REASON_EV, REASON_MAX_BETS,
        )
    }
    n_bets = int((work['skip_reason'] == REASON_BET).sum())
    logger.warning('購入条件: %s', rule.describe())
    logger.warning(
        '選別結果: %s 頭 / %s レース → 購入 %s 点'
        '（内訳: 総投票額 %s / オッズ下限 %s / オッズ上限 %s /'
        ' ドリフト %s / EV %s / レース内2位以下 %s）',
        f'{n:,}', f'{work["race_id"].nunique():,}', f'{n_bets:,}',
        counts[REASON_POOL], counts[REASON_ODDS_LOW], counts[REASON_ODDS_HIGH],
        counts[REASON_DRIFT], counts[REASON_EV], counts[REASON_MAX_BETS],
    )


def rule_from_config(config: dict) -> Tuple[WinRule, Dict[str, object]]:
    """``config.yaml`` の ``strategy`` セクションからルールを作る。

    Args:
        config: 全体設定辞書

    Returns:
        Tuple[WinRule, Dict[str, object]]: (ルール, 使った設定値)
    """
    cfg = config.get('strategy', {}) or {}
    rule = WinRule(
        min_ev=float(cfg.get('min_ev', DEFAULT_MIN_EV)),
        odds_min=float(cfg.get('odds_min', DEFAULT_ODDS_MIN)),
        odds_max=float(cfg.get('odds_max', DEFAULT_ODDS_MAX)),
        max_drift=float(cfg.get('max_drift', DEFAULT_MAX_DRIFT)),
        min_pool_yen=float(cfg.get('min_pool_yen', DEFAULT_MIN_POOL_YEN)),
        min_pool_pctile=(
            float(cfg['min_pool_pctile'])
            if cfg.get('min_pool_pctile') is not None
            else DEFAULT_MIN_POOL_PCTILE
        ),
        max_bets_per_race=int(
            cfg.get('max_bets_per_race', DEFAULT_MAX_BETS_PER_RACE)
        ),
        stake_yen=float(cfg.get('stake_yen', DEFAULT_STAKE_YEN)),
    )
    return rule, dict(cfg)
