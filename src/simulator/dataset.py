"""学習ラベル（着順）を ``data/processed/horses`` から読み込むモジュール。"""
import logging
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


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
