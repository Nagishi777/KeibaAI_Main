"""当日スクレイパが出力したレース別オッズ CSV を読み込むモジュール。

:mod:`src.scraping.realtime_odds` はレース・券種ごとに1ファイルを書き、
10s〜60m の全時点を ``snapshot_label`` 列で持つ。

    data/processed/realtime_odds/{YYYYMMDD}/{race_id}_tansho_realtimeodds.csv

本モジュールは当日フォルダの単勝ファイルをすべて読み、10m と 5m の行を
``[race_id, horse_number, odds_10m_raw, pool_10m_raw, odds_5m_raw, pool_5m_raw]``
という :func:`src.simulator.features.attach_features` が期待する
ワイド表に整形する。

学習側が読む月別 CSV（``data/processed/odds_series``）とは列名・ファイル
構成が違うだけで、意味は同じ（``odds_win`` = 単勝オッズ / ``hyosu_total`` =
レース総票数・百円単位）。整形後は同じ関数で特徴量を作るため、当日と
過去で特徴量の定義がずれることはない。

IMPORTANT (欠損を補完しない):
    10m または 5m のオッズが取れていないレースは**予測対象から外す**。
    片方の値で代用すると ``inc_share_10m_5m`` が 0 になり
    「資金が動かなかった」と誤って断定することになる。
"""
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from src.simulator.features import REQUIRED_SNAPSHOTS

logger = logging.getLogger(__name__)

# 当日スクレイパの出力先（realtime_odds.DEFAULT_OUTPUT_DIR と同じ）
DEFAULT_REALTIME_DIR = Path('data/processed/realtime_odds')

# CSV は BOM 付き UTF-8 で保存される
_ENCODING = 'utf-8-sig'

# 読み込む券種（単勝のみ）
BET_TYPE = 'tansho'

# レース別 CSV のファイル名接尾辞（realtime_odds.race_odds_path と同じ規則）
_FILE_SUFFIX = f'_{BET_TYPE}_realtimeodds.csv'

# CSV 側の列名 → 整形後の列名
_VALUE_COLS = {'odds_win': 'odds', 'hyosu_total': 'pool'}


def date_dir(realtime_dir: Path, target_date: dt.date) -> Path:
    """対象日のフォルダ ``realtime_dir/YYYYMMDD`` を返す。

    Args:
        realtime_dir: ``data/processed/realtime_odds`` のパス
        target_date: 対象日

    Returns:
        Path: 対象日のフォルダ
    """
    return Path(realtime_dir) / f'{target_date:%Y%m%d}'


def race_files(
    realtime_dir: Path,
    target_date: dt.date,
    race_ids: Optional[Iterable[str]] = None,
) -> List[Path]:
    """対象日の単勝レース別 CSV を列挙する。

    Args:
        realtime_dir: ``data/processed/realtime_odds`` のパス
        target_date: 対象日
        race_ids: 指定したレースのファイルだけに絞る（省略時は全レース）

    Returns:
        List[Path]: ``{race_id}_tansho_realtimeodds.csv`` のパス（名前順）

    Raises:
        FileNotFoundError: 対象のファイルが1件も無い場合
    """
    folder = date_dir(realtime_dir, target_date)
    if race_ids is None:
        paths = sorted(folder.glob(f'*{_FILE_SUFFIX}'))
    else:
        candidates = (folder / f'{race_id}{_FILE_SUFFIX}' for race_id in set(race_ids))
        paths = sorted(path for path in candidates if path.exists())
    if not paths:
        raise FileNotFoundError(
            f'当日のオッズ CSV が見つかりません: {folder}/*{_FILE_SUFFIX}。'
            ' 先に python -m src.scraping.realtime_odds を'
            ' 実行して当日のオッズを取得してください。'
        )
    return paths


def _read_race_files(paths: List[Path]) -> pd.DataFrame:
    """レース別 CSV をすべて読み、1つの縦長表にまとめる。

    Args:
        paths: :func:`race_files` の戻り値

    Returns:
        pd.DataFrame: 全レース・全時点の行

    Raises:
        ValueError: 必要な列が欠けている場合
    """
    need = ['race_id', 'umaban', 'snapshot_label'] + list(_VALUE_COLS)
    frames = []
    for path in paths:
        df = pd.read_csv(
            path, encoding=_ENCODING,
            dtype={'race_id': str, 'umaban': str, 'snapshot_label': str},
            low_memory=False,
        )
        missing = [c for c in need if c not in df.columns]
        if missing:
            raise ValueError(f'{path.name} に必要な列がありません: {missing}')
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def _read_snapshot(rows: pd.DataFrame, label: str) -> pd.DataFrame:
    """縦長表から1時点分を取り出し、馬単位の最新行だけを残す。

    同一 ``(race_id, umaban)`` が複数回保存されている場合
    （``target_datetime`` 違い）は、発表時刻が最も新しい行を採る。

    Args:
        rows: :func:`_read_race_files` の戻り値
        label: 時点ラベル（列名の接尾辞に使う）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_{label}_raw, pool_{label}_raw]

    Raises:
        ValueError: 馬番を数値として解釈できない行がある場合
    """
    df = rows[rows['snapshot_label'] == label].copy()

    # 同一馬が複数時刻で保存されていれば最新の発表値を採る
    if 'happyo_datetime' in df.columns:
        df['_happyo'] = pd.to_datetime(df['happyo_datetime'], errors='coerce')
        df = (
            df.sort_values('_happyo', kind='stable')
            .drop_duplicates(['race_id', 'umaban'], keep='last')
            .drop(columns='_happyo')
        )
    else:
        df = df.drop_duplicates(['race_id', 'umaban'], keep='last')

    horse_number = pd.to_numeric(df['umaban'], errors='coerce')
    n_bad = int(horse_number.isna().sum())
    if n_bad:
        raise ValueError(
            f'{label} 時点: 馬番を数値として解釈できない行が {n_bad} 件あります。'
            ' 補完せず処理を停止します。'
        )

    out = pd.DataFrame({
        'race_id': df['race_id'].astype(str),
        'horse_number': horse_number.astype('int64'),
    })
    for src_col, prefix in _VALUE_COLS.items():
        value = pd.to_numeric(df[src_col], errors='coerce')
        # オッズは正の倍率、票数合計は非負のみ有効。範囲外は NaN（補完しない）。
        bound = value > 0 if prefix == 'odds' else value >= 0
        out[f'{prefix}_{label}_raw'] = value.where(bound).astype('float64')

    logger.info(
        '%s 時点を読込: %s 行 / %s レース',
        label, f'{len(out):,}', f'{out["race_id"].nunique():,}',
    )
    return out


def load_today_snapshots(
    target_date: dt.date,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    *,
    race_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """当日の 10m / 5m を結合してワイド表にする。

    Args:
        target_date: 対象日
        realtime_dir: ``data/processed/realtime_odds`` のパス
        race_ids: 指定したレースだけを読む（省略時は全レース）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_5m_raw, pool_5m_raw,
        odds_10m_raw, pool_10m_raw]

    Raises:
        FileNotFoundError: 対象日の CSV が無い場合
        ValueError: 10m と 5m が1行も対応しない場合
    """
    base, decision = REQUIRED_SNAPSHOTS
    rows = _read_race_files(race_files(realtime_dir, target_date, race_ids))
    frames: Dict[str, pd.DataFrame] = {
        label: _read_snapshot(rows, label) for label in REQUIRED_SNAPSHOTS
    }

    # 両時点がそろった馬だけを対象にする。片方しか無いレースは inc_share を
    # 作れないため、代用せず落とす（「動かなかった」と誤断定しないため）。
    out = frames[decision].merge(
        frames[base], on=['race_id', 'horse_number'], how='inner'
    )
    if not len(out):
        raise ValueError(
            f'{base} と {decision} のオッズが1行も対応しませんでした。'
            f' {date_dir(realtime_dir, target_date)} の当日ファイルを確認してください。'
        )

    only_base = len(frames[base]) - len(out)
    only_decision = len(frames[decision]) - len(out)
    if only_base or only_decision:
        logger.warning(
            '片方の時点にしか存在しない行を除外: %sのみ %s 行 / %sのみ %s 行',
            base, f'{only_base:,}', decision, f'{only_decision:,}',
        )

    out = out.sort_values(['race_id', 'horse_number']).reset_index(drop=True)
    logger.warning(
        '当日オッズ読込: %s 行 / %s レース（%s）',
        f'{len(out):,}', f'{out["race_id"].nunique():,}', target_date,
    )
    return out


def available_dates(realtime_dir: Path = DEFAULT_REALTIME_DIR) -> List[dt.date]:
    """単勝のレース別 CSV が存在する日付を新しい順に列挙する（補助用）。

    Args:
        realtime_dir: ``data/processed/realtime_odds`` のパス

    Returns:
        List[dt.date]: 取得済みの日付（新しい順）
    """
    root = Path(realtime_dir)
    if not root.is_dir():
        return []
    dates: List[dt.date] = []
    for folder in root.iterdir():
        if not folder.is_dir() or not any(folder.glob(f'*{_FILE_SUFFIX}')):
            continue
        try:
            dates.append(dt.datetime.strptime(folder.name, '%Y%m%d').date())
        except ValueError:
            continue
    return sorted(dates, reverse=True)
