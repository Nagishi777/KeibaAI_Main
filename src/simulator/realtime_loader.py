"""当日スクレイパが出力した時点別オッズ CSV を読み込むモジュール。

:mod:`src.simulator.scraping.realtime_odds` は時点ごとに別ファイルへ書く。

    data/processed/realtime_odds/{YYYYMMDD}_tansho_{label}.csv

本モジュールは 5m と 1m の2ファイルを読み、
``[race_id, horse_number, odds_5m_raw, pool_5m_raw, odds_1m_raw, pool_1m_raw]``
という :func:`src.simulator.features.attach_features` が期待する
ワイド表に整形する。

学習側が読む月別 CSV（``data/processed/odds_series``）とは列名・ファイル
構成が違うだけで、意味は同じ（``odds_win`` = 単勝オッズ / ``hyosu_total`` =
レース総票数・百円単位）。整形後は同じ関数で特徴量を作るため、当日と
過去で特徴量の定義がずれることはない。

IMPORTANT (欠損を補完しない):
    1m のファイルが無い、または 1m のオッズが取れていないレースは
    **予測対象から外す**。5m の値で代用すると ``inc_share_5m_1m`` が 0 に
    なり「直前に資金が動かなかった」と誤って断定することになる。
"""
import datetime as dt
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# 当日スクレイパの出力先（realtime_odds.DEFAULT_OUTPUT_DIR と同じ）
DEFAULT_REALTIME_DIR = Path('data/processed/realtime_odds')

# CSV は BOM 付き UTF-8 で保存される
_ENCODING = 'utf-8-sig'

# 読み込む券種（単勝のみ）
BET_TYPE = 'tansho'

# 必要な時点と、整形後の列接頭辞
SNAPSHOT_LABELS: Tuple[str, str] = ('5m', '1m')

# CSV 側の列名 → 整形後の列名
_VALUE_COLS = {'odds_win': 'odds', 'hyosu_total': 'pool'}


def snapshot_path(realtime_dir: Path, target_date: dt.date, label: str) -> Path:
    """時点別 CSV のパスを組み立てる。

    Args:
        realtime_dir: ``data/processed/realtime_odds`` のパス
        target_date: 対象日
        label: 時点ラベル（'5m' / '1m'）

    Returns:
        Path: CSV のパス
    """
    return Path(realtime_dir) / f'{target_date:%Y%m%d}_{BET_TYPE}_{label}.csv'


def _read_snapshot(path: Path, label: str) -> pd.DataFrame:
    """時点別 CSV を1件読み、馬単位の最新行だけを取り出す。

    同一 ``(race_id, umaban)`` が複数回保存されている場合
    （``target_datetime`` 違い）は、発表時刻が最も新しい行を採る。

    Args:
        path: CSV のパス
        label: 時点ラベル（列名の接尾辞に使う）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_{label}_raw, pool_{label}_raw]

    Raises:
        FileNotFoundError: CSV が無い場合
        ValueError: 必要な列が欠けている場合
    """
    if not path.exists():
        raise FileNotFoundError(
            f'{label} 時点のオッズ CSV が見つかりません: {path}。'
            ' 先に python -m src.simulator.scraping.realtime_odds を'
            ' 実行して当日のオッズを取得してください。'
        )

    df = pd.read_csv(path, encoding=_ENCODING, dtype={'race_id': str}, low_memory=False)
    need = ['race_id', 'umaban'] + list(_VALUE_COLS)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f'{path.name} に必要な列がありません: {missing}')

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
            f'{path.name}: 馬番を数値として解釈できない行が {n_bad} 件あります。'
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
        '%s 時点を読込: %s（%s 行 / %s レース）',
        label, path.name, f'{len(out):,}', f'{out["race_id"].nunique():,}',
    )
    return out


def load_today_snapshots(
    target_date: dt.date,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
) -> pd.DataFrame:
    """当日の 5m / 1m を結合してワイド表にする。

    Args:
        target_date: 対象日
        realtime_dir: ``data/processed/realtime_odds`` のパス

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_5m_raw, pool_5m_raw,
        odds_1m_raw, pool_1m_raw]

    Raises:
        FileNotFoundError: どちらかの時点の CSV が無い場合
        ValueError: 5m と 1m が1行も対応しない場合
    """
    frames: Dict[str, pd.DataFrame] = {
        label: _read_snapshot(snapshot_path(realtime_dir, target_date, label), label)
        for label in SNAPSHOT_LABELS
    }

    # 1m にある馬だけを対象にする。1m が無いレースは inc_share を作れないため、
    # 5m で代用せず落とす（「直前に動かなかった」と誤断定しないため）。
    out = frames['1m'].merge(frames['5m'], on=['race_id', 'horse_number'], how='inner')
    if not len(out):
        raise ValueError(
            '5m と 1m のオッズが1行も対応しませんでした。'
            f' {realtime_dir} の当日ファイルを確認してください。'
        )

    only_5m = len(frames['5m']) - len(out)
    only_1m = len(frames['1m']) - len(out)
    if only_5m or only_1m:
        logger.warning(
            '片方の時点にしか存在しない行を除外: 5mのみ %s 行 / 1mのみ %s 行',
            f'{only_5m:,}', f'{only_1m:,}',
        )

    out = out.sort_values(['race_id', 'horse_number']).reset_index(drop=True)
    logger.warning(
        '当日オッズ読込: %s 行 / %s レース（%s）',
        f'{len(out):,}', f'{out["race_id"].nunique():,}', target_date,
    )
    return out


def available_dates(realtime_dir: Path = DEFAULT_REALTIME_DIR) -> List[dt.date]:
    """1m ファイルが存在する日付を新しい順に列挙する（補助用）。

    Args:
        realtime_dir: ``data/processed/realtime_odds`` のパス

    Returns:
        List[dt.date]: 取得済みの日付（新しい順）
    """
    dates: List[dt.date] = []
    for path in Path(realtime_dir).glob(f'*_{BET_TYPE}_1m.csv'):
        stamp = path.name.split('_', 1)[0]
        try:
            dates.append(dt.datetime.strptime(stamp, '%Y%m%d').date())
        except ValueError:
            continue
    return sorted(dates, reverse=True)
