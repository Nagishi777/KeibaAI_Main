"""スタッキングの専門家（base learner）定義.

``docs/report/20260811_market_edge_analysis.md`` §5 P1 に基づく多専門家構成。

設計方針（なぜ「1特徴量ファミリー = 1モデル」にしないか）:
    特徴量を分割しても情報は増えず、**交互作用が壊れる**。540列を見る1本の木は
    「血統 × 馬場状態」を学習できるが、分割された2モデルにはできない。
    スタッキングが効くのは各専門家が (i) 単体で有能かつ
    (ii) **誤差が相関しない** 場合に限られる。

    したがって血統専門家は作らない。血統は約32列でほぼ静的なため、
    単独モデルはレース内でほぼ定数に潰れ、かつ誤差が E1 と強く相関する
    （E1 は同じ32列を距離・馬場との交互作用込みで既に使っている）。
    非相関化の利得がゼロで、分散とキャリブレーション対象だけが増える。

専門家一覧:
    E1 能力     : 市場列を除く全特徴量。市場独立性が Meta の残差チャネルの前提
    E2 ランキング: E1と同一特徴量・LambdaRank。同じ入力/異なる損失で誤差が非相関
    E3 調教     : 坂路+ウッド。市場が処理しにくい情報（レポート §4 ★★☆）
    E5a 市場残差: logit(p_market) を init_score にした条件適性モデル
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ExpertSpec:
    """1専門家の学習仕様.

    Attributes:
        expert_id: 専門家ID（モデルファイル名・スキーマ名に使う）
        feature_group: ``config/features.json`` の ``feature_groups`` のキー
        target_col: 目的変数の列名
        objective: 'binary' / 'lambdarank' / 'binary_residual'
        use_market_init_score: True なら ``logit(p_market)`` を init_score に使う
            （残差学習）。``objective='binary_residual'`` と対で使う
        min_date: この日付以降のデータのみで学習する（データ可用性の制約）
        min_date_reason: ``min_date`` を設ける理由（監査用）
        calibrate: 学習後に isotonic キャリブレーションを行うか
        extra_feature_groups: 追加で結合する特徴量グループ
    """

    expert_id: str
    feature_group: str
    target_col: str
    objective: str
    use_market_init_score: bool = False
    min_date: Optional[str] = None
    min_date_reason: str = ''
    calibrate: bool = True
    extra_feature_groups: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        valid = {'binary', 'lambdarank', 'binary_residual'}
        if self.objective not in valid:
            raise ValueError(
                f"未知の objective です: '{self.objective}'（有効: {sorted(valid)}）"
            )
        if self.use_market_init_score and self.objective != 'binary_residual':
            raise ValueError(
                f"[{self.expert_id}] use_market_init_score=True には "
                "objective='binary_residual' が必要です"
            )
        if self.objective == 'binary_residual' and not self.use_market_init_score:
            raise ValueError(
                f"[{self.expert_id}] objective='binary_residual' には "
                "use_market_init_score=True が必要です"
            )


# --- 専門家ロースター -------------------------------------------------------

E1_ABILITY = ExpertSpec(
    expert_id='e1_ability',
    feature_group='ability',
    target_col='target_win',
    objective='binary',
)

E2_RANKING = ExpertSpec(
    expert_id='e2_ranking',
    feature_group='ability',
    target_col='finish_position',
    objective='lambdarank',
    calibrate=False,  # ランキングスコアは確率ではないためキャリブレートしない
)

E3_TRAINING = ExpertSpec(
    expert_id='e3_training',
    feature_group='training',
    target_col='target_win',
    objective='binary',
    min_date='2022-01-01',
    min_date_reason=(
        'ウッド調教データのカバレッジが2021年は17.3%、2022年から88.5%、'
        '2023年以降95%+。2021年を含めると wood_data_available が'
        '年代の代理変数になるため'
    ),
)

E5A_MARKET = ExpertSpec(
    expert_id='e5a_market',
    feature_group='condition_fit',
    target_col='target_win',
    objective='binary_residual',
    use_market_init_score=True,
    extra_feature_groups=('market',),
)

#: 既定のロースター。Stage 2 は E1/E2/E5a、Stage 3 で E3 を追加する。
DEFAULT_ROSTER: tuple[ExpertSpec, ...] = (E1_ABILITY, E2_RANKING, E5A_MARKET)

#: expert_id → ExpertSpec
ALL_EXPERTS: dict[str, ExpertSpec] = {
    e.expert_id: e for e in (E1_ABILITY, E2_RANKING, E3_TRAINING, E5A_MARKET)
}


def resolve_roster(expert_ids: list[str]) -> list[ExpertSpec]:
    """設定の expert_id リストから ExpertSpec のリストを解決する.

    Args:
        expert_ids: ``config.model.ensemble.experts`` の値

    Returns:
        list[ExpertSpec]: 指定順の専門家仕様

    Raises:
        ValueError: 未知の expert_id が含まれる場合、またはリストが空の場合
    """
    if not expert_ids:
        raise ValueError(
            'ensemble.experts が空です。最低1つの専門家を指定してください。'
        )
    unknown = [e for e in expert_ids if e not in ALL_EXPERTS]
    if unknown:
        raise ValueError(
            f'未知の expert_id です: {unknown}（有効: {sorted(ALL_EXPERTS)}）'
        )
    return [ALL_EXPERTS[e] for e in expert_ids]
