"""当日スクレイパが出力したレース別オッズ CSV を読み込むモジュール。

:mod:`src.scraping.realtime_odds` はレース・券種ごとに1ファイルを書き、
10s〜60m の全時点を ``snapshot_label`` 列で持つ。

    data/processed/realtime_odds/{YYYYMMDD}/{race_id}_tansho_realtimeodds.csv

本モジュールは当日フォルダの単勝ファイルをすべて読み、穴馬戦略が必要とする
時点（``60m / 30m / 10m / 5m`` と判断時点 ``2m``）を
``odds_{snap}_raw`` / ``pool_{snap}_raw`` というワイド表に整形する。
整形後は :mod:`src.simulator.features` の同じ関数で特徴量を作るため、
当日と過去で特徴量の定義がずれることはない。

特徴量は市場スナップショットのみ（22列）なので、馬の属性は読まない。
理由は :mod:`src.simulator.features` の冒頭の NOTE を参照
（当日は出馬表から馬体重・斤量が取得できず、学習と当日で入力がずれるため
特徴量から外している）。

IMPORTANT (欠損を補完しない):
    判断時点（2m）または 5m のオッズが取れていないレースは**予測対象から
    外す**。片方の値で代用すると ``drift_5_late`` が 0 になり
    「資金が動かなかった」と誤って断定することになる。
"""
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src.simulator.features import LIVE_DECISION_SNAPSHOT, required_snapshots

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

# 判断時点が欠けていたときに代用を許す時点（発走に近い順）。
# 2m が取れていないレースを黙って捨てないための代替であり、
# 使った場合は警告を出して CSV にも実際の時点を残す。
FALLBACK_SNAPSHOTS: tuple[str, ...] = ('3m', '4m')


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


def _read_snapshot(rows: pd.DataFrame, label: str, out_label: str) -> pd.DataFrame:
    """縦長表から1時点分を取り出し、馬単位の最新行だけを残す。

    同一 ``(race_id, umaban)`` が複数回保存されている場合
    （``target_datetime`` 違い）は、発表時刻が最も新しい行を採る。

    Args:
        rows: :func:`_read_race_files` の戻り値
        label: CSV 上の時点ラベル
        out_label: 出力列名に使う時点ラベル（代用時は元の時点と異なる）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_{out_label}_raw,
        pool_{out_label}_raw]

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
        out[f'{prefix}_{out_label}_raw'] = value.where(bound).astype('float64')

    logger.info(
        '%s 時点を読込: %s 行 / %s レース',
        label, f'{len(out):,}', f'{out["race_id"].nunique():,}',
    )
    return out


def _read_decision_snapshot(
    rows: pd.DataFrame, decision_snapshot: str
) -> pd.DataFrame:
    """判断時点を読む。取れていないレースは直後の時点で代用する。

    2m の取得に失敗したレースを黙って捨てると、その日の購入機会が理由なく
    消える。よって 3m → 4m の順で代用を試み、使った時点を
    ``decision_snapshot_used`` 列に残す。

    Args:
        rows: :func:`_read_race_files` の戻り値
        decision_snapshot: 本来使う判断時点（``2m``）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_{decision}_raw,
        pool_{decision}_raw, decision_snapshot_used]
    """
    primary = _read_snapshot(rows, decision_snapshot, decision_snapshot)
    primary['decision_snapshot_used'] = decision_snapshot
    have = set(primary.loc[primary[f'odds_{decision_snapshot}_raw'].notna(), 'race_id'])

    extras: List[pd.DataFrame] = []
    for fallback in FALLBACK_SNAPSHOTS:
        missing_rows = rows[~rows['race_id'].isin(have)]
        if not len(missing_rows) or fallback not in set(rows['snapshot_label']):
            continue
        alt = _read_snapshot(missing_rows, fallback, decision_snapshot)
        alt = alt[alt[f'odds_{decision_snapshot}_raw'].notna()]
        if not len(alt):
            continue
        alt['decision_snapshot_used'] = fallback
        logger.warning(
            '%s 時点が取れていない %s レースを %s 時点で代用します: %s',
            decision_snapshot, f'{alt["race_id"].nunique():,}', fallback,
            ','.join(sorted(set(alt['race_id']))),
        )
        extras.append(alt)
        have |= set(alt['race_id'])

    if not extras:
        return primary
    primary = primary[primary[f'odds_{decision_snapshot}_raw'].notna()]
    return pd.concat([primary] + extras, ignore_index=True, sort=False)


def load_today_snapshots(
    target_date: dt.date,
    realtime_dir: Path = DEFAULT_REALTIME_DIR,
    *,
    race_ids: Optional[Iterable[str]] = None,
    decision_snapshot: str = LIVE_DECISION_SNAPSHOT,
) -> pd.DataFrame:
    """当日の各時点を結合してワイド表にする。

    Args:
        target_date: 対象日
        realtime_dir: ``data/processed/realtime_odds`` のパス
        race_ids: 指定したレースだけを読む（省略時は全レース）
        decision_snapshot: 判断時点（既定は ``2m``）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_{snap}_raw,
        pool_{snap}_raw（60m/30m/10m/5m/判断時点）, decision_snapshot_used]

    Raises:
        FileNotFoundError: 対象日の CSV が無い場合
        ValueError: 5m と判断時点が1行も対応しない場合
    """
    snaps = required_snapshots(decision_snapshot)
    rows = _read_race_files(race_files(realtime_dir, target_date, race_ids))

    # 判断時点は代用を許す。他の時点はそのまま読む。
    out = _read_decision_snapshot(rows, decision_snapshot)
    for label in snaps:
        if label == decision_snapshot:
            continue
        frame = _read_snapshot(rows, label, label)
        # 5m は判断時点と対でドリフトを作るので内部結合（片方だけでは意味が無い）
        how = 'inner' if label == '5m' else 'left'
        before = len(out)
        out = out.merge(frame, on=['race_id', 'horse_number'], how=how)
        if how == 'inner' and len(out) < before:
            logger.warning(
                '%s 時点が無い %s 行を除外しました（ドリフトを作れないため）',
                label, f'{before - len(out):,}',
            )

    if not len(out):
        raise ValueError(
            f'5m と {decision_snapshot} のオッズが1行も対応しませんでした。'
            f' {date_dir(realtime_dir, target_date)} の当日ファイルを確認してください。'
        )

    out = out.sort_values(['race_id', 'horse_number']).reset_index(drop=True)
    used = out['decision_snapshot_used'].value_counts().to_dict()
    logger.warning(
        '当日オッズ読込: %s 行 / %s レース（%s / 判断時点の内訳 %s）',
        f'{len(out):,}', f'{out["race_id"].nunique():,}', target_date, used,
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
