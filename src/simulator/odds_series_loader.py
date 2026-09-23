"""JV-Link 時系列オッズ（``data/processed/odds_series``）の共通ローダ。

``src/scraping/odds_series_scraper.py`` が月別に出力した
``YYYYMM_odds_series_tansho.csv`` を読み込み、
``[race_id, horse_number]`` をキーとする馬単位のワイドテーブルへ整形する。

``market_odds_features`` / ``market_votes_features`` の双方がこのモジュールを
参照し、CSV の読み込み・期間フィルタ・スナップショット選択を一元化する。

IMPORTANT (データリーク防止):
    扱うのは**レース確定前**の市場スナップショットのみである。
    確定単勝オッズ ``odds_win`` / 確定人気 ``popularity`` / 着順 / 払戻は
    このモジュールから一切参照しない。

IMPORTANT (スナップショットの信頼性):
    実データ調査（2016-2026 / 約3.6万レース）の結果は次の通り。

    ========  ===============================================
    区間      直前スナップショットと票数が同一のレース比率
    ========  ===============================================
    30m→10m   全年 0.0%
    10m→5m    全年 0.0%
    5m→1m     2016-2021 約98% / 2022 66% / 2023以降 約3%
    ========  ===============================================

    ``1m`` は 2022年以前ではほぼ更新が取れておらず、これを既定の特徴量に
    含めると「年代の代理変数」になってしまう。よって既定のスナップショットは
    ``60m/30m/10m/5m`` の4点とし、``1m`` は ``--include-1m`` を明示した
    場合のみ追加する。``1m`` を含める場合も停滞レースの増分は NaN とし、
    0 で補完して「動きが無かった」と誤って断定することはしない。
"""
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CSV は BOM 付き UTF-8 で保存されている
_ENCODING = 'utf-8-sig'

# 月別ファイル名のパターン（券種は単勝のみ使う）
_FILENAME_RE = re.compile(r'^(?P<period>\d{6})_odds_series_(?P<bet_type>\w+)\.csv$')

# 既定で使うスナップショット（時系列順）。1m は停滞が激しいため既定から外す。
FEATURE_SNAPSHOTS: tuple[str, ...] = ('60m', '30m', '10m', '5m')

# ``--include-1m`` 指定時に末尾へ追加するスナップショット
LATE_SNAPSHOT: str = '1m'

# CSV に存在する全スナップショット
ALL_SNAPSHOTS: tuple[str, ...] = ('60m', '30m', '10m', '5m', '3m', '1m')

# 結合キー
KEY_COLS: tuple[str, str] = ('race_id', 'horse_number')

# 単勝の JRA 控除率（票数の逆算に使う）
TAKEOUT_RATE: float = 0.20

# 0除算・log(0) を避ける下限
EPS: float = 1e-12


def resolve_snapshots(include_1m: bool = False) -> tuple[str, ...]:
    """使用するスナップショット列を決定する。

    Args:
        include_1m: True なら末尾に ``1m`` を追加する

    Returns:
        tuple[str, ...]: スナップショット名（時系列順）
    """
    return FEATURE_SNAPSHOTS + ((LATE_SNAPSHOT,) if include_1m else ())


def _iter_period_files(
    odds_series_dir: Path,
    start_period: str,
    end_period: str,
    bet_type: str = 'tansho',
) -> List[Path]:
    """期間に該当する月別 CSV を年月昇順で列挙する。

    Args:
        odds_series_dir: ``data/processed/odds_series`` のパス
        start_period: 開始年月 (YYYYMM)
        end_period: 終了年月 (YYYYMM)
        bet_type: 券種（既定 ``tansho``）

    Returns:
        List[Path]: 該当ファイルのパス（年月昇順）

    Raises:
        NotADirectoryError: ディレクトリが存在しない場合
        ValueError: 期間指定が不正、または該当ファイルが1件も無い場合
    """
    if not odds_series_dir.is_dir():
        raise NotADirectoryError(
            f'時系列オッズディレクトリが見つかりません: {odds_series_dir}'
        )
    for label, value in (('start_period', start_period), ('end_period', end_period)):
        if not re.fullmatch(r'\d{6}', str(value)):
            raise ValueError(f'{label} は YYYYMM 形式で指定してください: {value}')
    if start_period > end_period:
        raise ValueError(
            f'start_period が end_period より後です: {start_period} > {end_period}'
        )

    paths: List[Path] = []
    for path in sorted(odds_series_dir.glob(f'*_odds_series_{bet_type}.csv')):
        matched = _FILENAME_RE.match(path.name)
        if matched is None:
            continue
        if start_period <= matched.group('period') <= end_period:
            paths.append(path)

    if not paths:
        raise ValueError(
            f'期間 {start_period}〜{end_period} に該当する時系列オッズ'
            f'（{bet_type}）が {odds_series_dir} にありません。'
            ' 先に src.scraping.odds_series_scraper を実行してください。'
        )
    logger.info(
        f'時系列オッズ対象ファイル: {len(paths)} 件'
        f'（{paths[0].name} 〜 {paths[-1].name}）'
    )
    return paths


def _read_one(path: Path, snapshots: Sequence[str]) -> pd.DataFrame:
    """月別 CSV を1件読み込み、必要列だけを取り出す。

    Args:
        path: 月別 CSV のパス
        snapshots: 使用するスナップショット名

    Returns:
        pd.DataFrame: [race_id, date, horse_number, odds_win_*, hyosu_total_*]

    Raises:
        ValueError: 必須列が欠けている、または馬番が数値化できない場合
    """
    base_cols = ['race_id', 'date', 'umaban']
    value_cols = [f'odds_win_{s}' for s in snapshots]
    total_cols = [f'hyosu_total_{s}' for s in snapshots]
    need = base_cols + value_cols + total_cols

    header = pd.read_csv(path, encoding=_ENCODING, nrows=0).columns.tolist()
    missing = [c for c in need if c not in header]
    if missing:
        raise ValueError(
            f'{path.name} に必要な列がありません: {missing}。'
            ' 時系列オッズ CSV の形式を確認してください。'
        )

    df = pd.read_csv(path, encoding=_ENCODING, usecols=need, low_memory=False)
    df = df.rename(columns={'umaban': 'horse_number'})
    df['race_id'] = df['race_id'].astype(str)
    horse_number = pd.to_numeric(df['horse_number'], errors='coerce')

    n_bad = int(horse_number.isna().sum())
    if n_bad:
        raise ValueError(
            f'{path.name}: 馬番を数値として解釈できない行が {n_bad} 件あります。'
            ' 補完せず処理を停止します。'
        )
    df['horse_number'] = horse_number.astype('int64')
    return df


def load_odds_series(
    odds_series_dir: Path | str,
    start_period: str,
    end_period: str,
    *,
    snapshots: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """指定期間の締切前単勝オッズを馬単位のワイド表として読み込む。

    Args:
        odds_series_dir: ``data/processed/odds_series`` のパス
        start_period: 開始年月 (YYYYMM)
        end_period: 終了年月 (YYYYMM)
        snapshots: 使用するスナップショット。省略時は ``FEATURE_SNAPSHOTS``

    Returns:
        pd.DataFrame: [race_id, date, horse_number,
        ``odds_{snap}_raw``…, ``pool_{snap}_raw``…]

    Raises:
        ValueError: 未知のスナップショット名、または
            ``race_id`` × ``horse_number`` が重複する場合
    """
    snaps = tuple(snapshots) if snapshots is not None else FEATURE_SNAPSHOTS
    unknown = [s for s in snaps if s not in ALL_SNAPSHOTS]
    if unknown:
        raise ValueError(
            f'サポートされていないスナップショットです: {unknown}'
            f'（利用可能: {list(ALL_SNAPSHOTS)}）'
        )

    paths = _iter_period_files(Path(odds_series_dir), start_period, end_period)
    frames = [_read_one(path, snaps) for path in paths]
    out = pd.concat(frames, ignore_index=True)

    dup = out.duplicated(subset=list(KEY_COLS))
    if bool(dup.any()):
        examples = out.loc[dup, list(KEY_COLS)].head(5).to_dict('records')
        raise ValueError(
            f'race_id × horse_number が重複しています（{int(dup.sum())} 行）:'
            f' 例 {examples}。補完せず処理を停止します。'
        )

    rename: Dict[str, str] = {}
    for snap in snaps:
        rename[f'odds_win_{snap}'] = f'odds_{snap}_raw'
        rename[f'hyosu_total_{snap}'] = f'pool_{snap}_raw'
    out = out.rename(columns=rename)

    for snap in snaps:
        # オッズは正の倍率のみ有効。0以下・欠損は NaN にする（補完しない）。
        odds = pd.to_numeric(out[f'odds_{snap}_raw'], errors='coerce')
        out[f'odds_{snap}_raw'] = odds.where(odds > 0).astype('float64')
        # 票数合計（百円単位）。負値はあり得ないので NaN にする。
        pool = pd.to_numeric(out[f'pool_{snap}_raw'], errors='coerce')
        out[f'pool_{snap}_raw'] = pool.where(pool >= 0).astype('float64')

    out = out.sort_values(['race_id', 'horse_number']).reset_index(drop=True)
    logger.info(
        f'締切前オッズ読込: {len(out)} 行 / {out["race_id"].nunique()} レース'
        f' / スナップショット {list(snaps)}'
    )
    return out


def estimate_votes(odds: np.ndarray, pool: np.ndarray) -> np.ndarray:
    """単勝オッズとレース総票数から馬番別の推定投票額を逆算する。

    JRA の単勝オッズは次式で決まる::

        odds_i = (1 - TAKEOUT_RATE) * 総投票額 / 投票額_i

    これを ``投票額_i`` について解く。``pool`` は百円単位の累計票数なので、
    戻り値も同じ百円単位になる。

    NOTE:
        オッズは小数第1位に丸めた表示値のため、各馬の推定票数の合計は
        ``(1 - TAKEOUT_RATE) * pool`` と丸め誤差の範囲でずれる。
        シェア・増分はいずれもレース内で再正規化するため影響は小さい。

    Args:
        odds: 単勝オッズの配列（非正・欠損は NaN であること）
        pool: レース総票数（百円単位）の配列

    Returns:
        np.ndarray: 推定投票額（百円単位、float64）。算出不能な行は NaN
    """
    valid = (odds > EPS) & (pool > EPS) & np.isfinite(odds) & np.isfinite(pool)
    with np.errstate(invalid='ignore', divide='ignore'):
        votes = np.where(
            valid,
            (1.0 - TAKEOUT_RATE) * np.where(valid, pool, 0.0)
            / np.where(valid, odds, 1.0),
            np.nan,
        )
    return votes.astype('float64')


def load_votes_series(
    odds_series_dir: Path | str,
    start_period: str,
    end_period: str,
    *,
    snapshots: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """指定期間の締切前オッズから馬番別の推定投票額を復元して読み込む。

    Args:
        odds_series_dir: ``data/processed/odds_series`` のパス
        start_period: 開始年月 (YYYYMM)
        end_period: 終了年月 (YYYYMM)
        snapshots: 使用するスナップショット。省略時は ``FEATURE_SNAPSHOTS``

    Returns:
        pd.DataFrame: [race_id, date, horse_number,
        ``votes_{snap}_raw``…, ``pool_{snap}_raw``…]
    """
    snaps = tuple(snapshots) if snapshots is not None else FEATURE_SNAPSHOTS
    out = load_odds_series(
        odds_series_dir, start_period, end_period, snapshots=snaps
    )
    for snap in snaps:
        out[f'votes_{snap}_raw'] = estimate_votes(
            out[f'odds_{snap}_raw'].to_numpy(dtype='float64'),
            out[f'pool_{snap}_raw'].to_numpy(dtype='float64'),
        )
    return out


def race_broadcast(
    df: pd.DataFrame, values: np.ndarray, how: str = 'sum'
) -> np.ndarray:
    """同一レース内の集約値を各行へブロードキャストする。

    Args:
        df: ``race_id`` 列を持つ DataFrame
        values: 行に対応する値の配列
        how: ``groupby.transform`` に渡す集約名（``sum`` / ``max`` など）

    Returns:
        np.ndarray: 行ごとのレース集約値（float64）
    """
    tmp = pd.DataFrame({'race_id': df['race_id'].to_numpy(), '_v': values})
    return tmp.groupby('race_id', sort=False)['_v'].transform(how).to_numpy(
        dtype='float64'
    )


def safe_share(value: np.ndarray, total: np.ndarray) -> np.ndarray:
    """シェア ``value / total`` を求める。``total`` が非正なら NaN。

    Args:
        value: 分子
        total: 分母（レース合計）

    Returns:
        np.ndarray: シェア（0〜1）。算出不能な行は NaN
    """
    valid = total > EPS
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(valid, value / np.where(valid, total, 1.0), np.nan)


def log_coverage(df: pd.DataFrame, cols: Iterable[str], tag: str) -> None:
    """特徴量ごとの有効行比率をログ出力する（欠損は補完しない方針の可視化）。

    Args:
        df: 特徴量テーブル
        cols: 対象カラム名
        tag: ログの接頭辞
    """
    n = len(df)
    if not n:
        logger.warning(f'[{tag}] 0 行')
        return
    for col in cols:
        n_valid = int(df[col].notna().sum())
        logger.info(
            f'[{tag}] {col:28s}: {n_valid:>8d}/{n} ({n_valid / n * 100:5.1f}%)'
        )
