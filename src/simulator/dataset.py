"""市場特徴量ファイルに学習ラベル・評価メタ列を結合するデータセット構築モジュール。

``data/feature/market_*.feather`` は締切前オッズ・投票額のみから生成される
「市場特徴量だけ」のファイルで、学習ラベル（``target_win`` 等）も
回収率評価に必要な払戻列も持たない。本モジュールはそれらを
``data/processed/horses``（着順）と ``src.models.odds_series``
（締切前オッズ・確定払戻）から補って学習可能な DataFrame を作る。

IMPORTANT (時系列リーク防止):
    ここで結合する ``odds_pre_win`` / ``payout_win`` / ``finish_position`` は
    すべて**評価専用のメタ列**であり、``feature_cols`` には含めない。
    特徴量はあくまで Feather ファイルが持つ市場特徴量のみとする。
"""
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from src.models.odds_series import attach_meta_columns

logger = logging.getLogger(__name__)

# 特徴量ではなくキー・ラベル・評価メタとして扱う列
KEY_COLS = ('race_id', 'date', 'horse_number')
LABEL_COLS = ('finish_position', 'target_win', 'target_place')

# 投票額特徴量のうち、締切前オッズの決定論的変換（またはそれに極めて近い量）である列。
# docs/market_feature_backtest_report.md §4.4 で実測した相関:
#   votes_share_5m         vs 市場確率(0.8/odds_5m)          相関 0.9998
#   votes_share_delta_*    vs 上記シェアの区間差              同上（shareの差分）
#   votes_inc_share_{a}_{b}: 区間増分のシェア（shareの派生量）
#   votes_entropy_5m       vs オッズから計算したエントロピー   相関 1.0000
#   votes_hhi_5m           vs オッズから計算したHHI            相関 1.0000
#   votes_inc_entropy_10m_5m: 増分資金シェアのエントロピー（entropyの派生量）
# これらはオッズ列と実質同一の情報であり、odds_votes に含めると同じ情報を
# 二重に与えるだけになる（レポート§4.2「和集合が部分集合に劣る」の一因）。
# votes_drift_* はシェアの差分だが2時点間の増分を含むためオッズ近似との相関が
# 0.95前後とやや弱く、除外対象に含めない（ユーザー確認済み）。
ODDS_CORRELATED_VOTES_COLS = (
    'votes_share_5m',
    'votes_share_delta_60m_5m',
    'votes_inc_share_60m_30m',
    'votes_inc_share_30m_10m',
    'votes_inc_share_10m_5m',
    'votes_entropy_5m',
    'votes_hhi_5m',
    'votes_inc_entropy_10m_5m',
)

def load_finish_positions(parsed_data_dir: Path, race_ids: pd.Series) -> pd.DataFrame:
    """``data/processed/horses`` から着順を読み込む。

    対象 race_id の年月（race_id 先頭4桁 = 年）だけを読むのではなく、
    ファイル名の年月で絞ると開催年と施行年がずれるケースを取りこぼすため、
    全ファイルを読んでから対象 race_id で絞る。

    Args:
        parsed_data_dir: ``data/processed`` のパス
        race_ids: 必要な race_id（int64）

    Returns:
        pd.DataFrame: [race_id, horse_number, finish_position]

    Raises:
        FileNotFoundError: horses CSV が1件も無い場合
    """
    horses_dir = Path(parsed_data_dir) / 'horses'
    paths = sorted(horses_dir.glob('*_horses.csv'))
    if not paths:
        raise FileNotFoundError(f"着順ファイルが見つかりません: {horses_dir}/*_horses.csv")

    wanted = set(race_ids.unique().tolist())
    frames: List[pd.DataFrame] = []
    for path in paths:
        chunk = pd.read_csv(
            path,
            usecols=['race_id', 'horse_number', 'finish_position'],
            dtype={'race_id': 'int64', 'horse_number': 'int64'},
        )
        chunk = chunk[chunk['race_id'].isin(wanted)]
        if len(chunk):
            frames.append(chunk)

    if not frames:
        raise FileNotFoundError(
            f"対象 race_id の着順が {horses_dir} に1件もありません"
        )
    horses = pd.concat(frames, ignore_index=True)
    return horses.drop_duplicates(subset=['race_id', 'horse_number'])


def build_dataset(
    feature_path: Path,
    config: dict,
    *,
    exclude_feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """市場特徴量ファイルにラベル・評価メタ列を結合した学習用データを作る。

    Args:
        feature_path: ``data/feature/market_*.feather`` のパス
        config: 全体設定辞書（``data`` / ``evaluation`` セクションを参照）
        exclude_feature_cols: 特徴量カラムから除外する列名（存在しない列名を
            指定してもエラーにはしない。ファイルによって列構成が違うため）。
            例: ``ODDS_CORRELATED_VOTES_COLS`` でオッズと決定論的に対応する
            投票額特徴量を除外する。

    Returns:
        Tuple[pd.DataFrame, List[str]]:
            (メタ列付き DataFrame, モデル入力に使う特徴量カラム)

    Raises:
        ValueError: 着順が結合できない行がある場合（補完せず停止する）、
            または除外指定の結果カラムが0列になった場合
    """
    feature_path = Path(feature_path)
    df = pd.read_feather(feature_path)
    logger.info("特徴量ファイル読込: %s（%s 行 / %s 列）", feature_path, len(df), df.shape[1])

    # race_id は odds_series / horses 側が int64 なので揃える
    df['race_id'] = df['race_id'].astype('int64')
    df['date'] = pd.to_datetime(df['date'])

    exclude = set(exclude_feature_cols or ())
    feature_cols = [
        c for c in df.columns
        if c not in KEY_COLS and c not in LABEL_COLS and c not in exclude
    ]
    if not feature_cols:
        raise ValueError(f"特徴量カラムが0列です: {feature_path}")
    if exclude:
        excluded_present = exclude & set(df.columns)
        if excluded_present:
            logger.info(
                "特徴量から除外: %s 列 %s", len(excluded_present), sorted(excluded_present)
            )
            df = df.drop(columns=list(excluded_present))

    parsed_data_dir = Path(config['data'].get('parsed_data_dir', 'data/processed'))
    horses = load_finish_positions(parsed_data_dir, df['race_id'])
    df = df.merge(horses, on=['race_id', 'horse_number'], how='left')

    missing = int(df['finish_position'].isna().sum())
    if missing:
        raise ValueError(
            f"着順が結合できない行が {missing} 行あります: {feature_path}. "
            "data/processed/horses の期間が特徴量ファイルを覆っているか確認してください"
        )
    df['finish_position'] = df['finish_position'].astype('int64')

    # 出走取消・除外・失格（負値）は学習・評価の対象外にする
    scratched = int((df['finish_position'] <= 0).sum())
    if scratched:
        logger.info("出走取消・除外・失格を除外: %s 行", scratched)
        df = df[df['finish_position'] > 0].reset_index(drop=True)

    df['target_win'] = (df['finish_position'] == 1).astype('int64')
    df['target_place'] = (
        (df['finish_position'] >= 1) & (df['finish_position'] <= 3)
    ).astype('int64')

    # 賭け判断＝締切前オッズ / 払戻＝確定払戻金（評価専用メタ列。特徴量には含めない）
    df = attach_meta_columns(df, config, bet_types=('win',))

    logger.info(
        "データセット構築完了: %s 行 / 特徴量 %s 列 / %s 〜 %s",
        f'{len(df):,}', len(feature_cols), df['date'].min().date(), df['date'].max().date(),
    )
    return df, feature_cols
