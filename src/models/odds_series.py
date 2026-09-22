"""
締切前オッズ時系列ローダ・確定払戻結合モジュール

``data/processed/odds/`` に置かれた締切前オッズ時系列 CSV を読み込み、
wide 形式（1行1レース・馬番が列）から long 形式（1行1頭 / 1行1ペア）へ変換する。
さらに ``data/processed/payouts/`` の確定払戻を結合し、
「賭け判断＝10分前オッズ / 払戻＝確定払戻金」の2系統オッズを実現するための
評価専用メタ列を組み立てる。

対象ファイル（いずれも UTF-8 BOM 付き）:
    - ``odds_series_list_tansho.csv``  単勝  ``odds_{T}_{馬番}``
    - ``odds_series_list_fukusho.csv`` 複勝  ``odds_{T}_{馬番}_low`` / ``_high``
    - ``odds_series_list_umaren.csv``  馬連  ``odds_{T}_{馬番a}_{馬番b}``

``{T}`` は締切前スナップショット時点で ``10m`` / ``5m`` / ``1m`` のいずれか。

IMPORTANT (データリーク防止):
    ここで生成する列は全て**評価専用のメタ列**であり、特徴量には含めない。
    ``config/features.json`` の ``exclude_cols`` にも登録して二重に担保する。
"""
import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CSV は BOM 付き UTF-8 で保存されているため utf-8-sig で読む
_ENCODING = 'utf-8-sig'

# 締切前スナップショット時点（列名プレフィックス）
SNAPSHOTS = ('10m', '5m', '1m')

# ファイル名（data.odds_series_dir 配下）
FILENAMES = {
    'tansho': 'odds_series_list_tansho.csv',
    'fukusho': 'odds_series_list_fukusho.csv',
    'umaren': 'odds_series_list_umaren.csv',
}

# 3ファイル共通のキー列
_KEY_COLS = (
    'src_file', 'race_id', 'ticket_type', 'date',
    'venue', 'kai', 'day', 'race_no', 'start_time',
)

# 単勝: odds_10m_5 → 馬番 5
_RE_TANSHO = re.compile(r'^odds_(10m|5m|1m)_(\d+)$')
# 複勝: odds_10m_5_low / odds_10m_5_high → 馬番 5
_RE_FUKUSHO = re.compile(r'^odds_(10m|5m|1m)_(\d+)_(low|high)$')
# 馬連: odds_10m_3_7 → ペア (3, 7)
_RE_UMAREN = re.compile(r'^odds_(10m|5m|1m)_(\d+)_(\d+)$')

# 出走可能な最大馬番。fukusho ファイルには馬番の実体を持たない
# `odds_10m_1` 〜 `odds_10m_36`（全行 NaN のパーサ残骸）が存在するため、
# 馬番の上限を超える平坦な列は読み捨てる。
MAX_HORSE_NUMBER = 18


def _validate_snapshot(snapshot: str) -> None:
    """スナップショット時点の指定値を検証する。

    Args:
        snapshot: '10m' / '5m' / '1m' のいずれか

    Raises:
        ValueError: サポート外の値が指定された場合
    """
    if snapshot not in SNAPSHOTS:
        raise ValueError(
            f"サポートされていないスナップショット: {snapshot}（{list(SNAPSHOTS)} のいずれか）"
        )


def _read_odds_csv(path: Path) -> pd.DataFrame:
    """締切前オッズ時系列 CSV を読み込む。

    Args:
        path: CSV ファイルパス

    Returns:
        pd.DataFrame: 読み込んだ DataFrame

    Raises:
        FileNotFoundError: ファイルが存在しない場合
    """
    if not path.exists():
        raise FileNotFoundError(
            f"締切前オッズ時系列ファイルが見つかりません: {path}"
        )
    df = pd.read_csv(path, encoding=_ENCODING, low_memory=False)
    if 'race_id' not in df.columns:
        raise ValueError(f"race_id 列が存在しません: {path}")
    return df


def _dedupe_by_race_id(df: pd.DataFrame, odds_cols: Sequence[str], path: Path) -> pd.DataFrame:
    """race_id の重複行を検証してから除去する。

    tansho ファイルは ``..._odds_series_raw.txt`` と ``..._odds_series_tansho_raw.txt``
    の2つの命名規則で同一レースが2行入っている。オッズ値が完全一致することを
    確認した上で1行に落とす。値が食い違う場合は補完・警告で握り潰さず停止する。

    Args:
        df: 読み込んだ DataFrame
        odds_cols: 一致を検証するオッズ列
        path: エラーメッセージ用のファイルパス

    Returns:
        pd.DataFrame: race_id が一意な DataFrame

    Raises:
        ValueError: 重複行のオッズ値が一致しない場合
    """
    dup_count = int(df.duplicated(subset='race_id').sum())
    if dup_count == 0:
        return df

    # 重複レースでオッズ値が食い違っていないかを検証する
    conflict = (df.groupby('race_id')[list(odds_cols)].nunique() > 1).any(axis=1)
    n_conflict = int(conflict.sum())
    if n_conflict > 0:
        conflict_ids = conflict[conflict].index.tolist()[:10]
        raise ValueError(
            f"race_id が重複し、かつオッズ値が一致しない行が {n_conflict} レースあります: "
            f"{path}（例: {conflict_ids}）。データを修正してください。"
        )

    deduped = df.drop_duplicates(subset='race_id', keep='first')
    logger.info(
        f"race_id 重複を除去: {path.name} {len(df)} 行 → {len(deduped)} 行 "
        f"（重複 {dup_count} 行はオッズ値一致を確認済み）"
    )
    return deduped


def _melt_by_horse(
    df: pd.DataFrame,
    col_map: Dict[str, int],
    value_name: str,
) -> pd.DataFrame:
    """馬番をキーとする wide 列を long 形式へ変換する。

    Args:
        df: wide 形式の DataFrame（race_id 列必須）
        col_map: 列名 → 馬番 のマッピング
        value_name: 出力する値列の名前

    Returns:
        pd.DataFrame: [race_id, horse_number, value_name] の DataFrame
    """
    sub = df[['race_id'] + list(col_map.keys())]
    long_df = sub.melt(id_vars='race_id', var_name='_col', value_name=value_name)
    long_df['horse_number'] = long_df['_col'].map(col_map).astype('int64')
    long_df = long_df.drop(columns='_col')
    # 出走頭数に満たない馬番は NaN。ここで落として実在する行だけを残す。
    long_df = long_df.dropna(subset=[value_name])
    long_df[value_name] = long_df[value_name].astype(float)
    return long_df.reset_index(drop=True)


def load_tansho_odds(path: Path | str, snapshot: str = '10m') -> pd.DataFrame:
    """単勝の締切前オッズを long 形式で読み込む。

    Args:
        path: ``odds_series_list_tansho.csv`` のパス
        snapshot: 締切前スナップショット時点（'10m' / '5m' / '1m'）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_pre_win] の DataFrame

    Raises:
        ValueError: 対象スナップショットの列が1つも存在しない場合
    """
    _validate_snapshot(snapshot)
    path = Path(path)
    df = _read_odds_csv(path)

    col_map: Dict[str, int] = {}
    for col in df.columns:
        m = _RE_TANSHO.match(col)
        if m and m.group(1) == snapshot:
            horse_number = int(m.group(2))
            if horse_number <= MAX_HORSE_NUMBER:
                col_map[col] = horse_number
    if not col_map:
        raise ValueError(f"単勝オッズ列（snapshot={snapshot}）が見つかりません: {path}")

    df = _dedupe_by_race_id(df, list(col_map.keys()), path)
    out = _melt_by_horse(df, col_map, 'odds_pre_win')
    logger.info(
        f"単勝締切前オッズ読込: {len(out)} 行 / {out['race_id'].nunique()} レース "
        f"(snapshot={snapshot})"
    )
    return out


def load_fukusho_odds(path: Path | str, snapshot: str = '10m') -> pd.DataFrame:
    """複勝の締切前オッズ（下限・上限）を long 形式で読み込む。

    複勝オッズはレンジで公示されるため ``_low`` / ``_high`` の2列ペアで持つ。
    サフィックスの無い平坦な列（``odds_10m_1`` 〜 ``odds_10m_36``）は
    全行 NaN のパーサ残骸なので読み捨てる。

    Args:
        path: ``odds_series_list_fukusho.csv`` のパス
        snapshot: 締切前スナップショット時点（'10m' / '5m' / '1m'）

    Returns:
        pd.DataFrame: [race_id, horse_number, odds_pre_place_low, odds_pre_place_high]

    Raises:
        ValueError: 対象スナップショットの列が1つも存在しない場合
    """
    _validate_snapshot(snapshot)
    path = Path(path)
    df = _read_odds_csv(path)

    low_map: Dict[str, int] = {}
    high_map: Dict[str, int] = {}
    for col in df.columns:
        m = _RE_FUKUSHO.match(col)
        if not m or m.group(1) != snapshot:
            continue
        horse_number = int(m.group(2))
        if horse_number > MAX_HORSE_NUMBER:
            continue
        if m.group(3) == 'low':
            low_map[col] = horse_number
        else:
            high_map[col] = horse_number
    if not low_map or not high_map:
        raise ValueError(f"複勝オッズ列（snapshot={snapshot}）が見つかりません: {path}")

    df = _dedupe_by_race_id(df, list(low_map.keys()) + list(high_map.keys()), path)
    low_df = _melt_by_horse(df, low_map, 'odds_pre_place_low')
    high_df = _melt_by_horse(df, high_map, 'odds_pre_place_high')
    out = low_df.merge(high_df, on=['race_id', 'horse_number'], how='outer')
    logger.info(
        f"複勝締切前オッズ読込: {len(out)} 行 / {out['race_id'].nunique()} レース "
        f"(snapshot={snapshot})"
    )
    return out


def load_umaren_odds(path: Path | str, snapshot: str = '10m') -> pd.DataFrame:
    """馬連の締切前オッズを long 形式で読み込む。

    Args:
        path: ``odds_series_list_umaren.csv`` のパス
        snapshot: 締切前スナップショット時点（'10m' / '5m' / '1m'）

    Returns:
        pd.DataFrame: [race_id, horse_number_a, horse_number_b, odds_pre_umaren]
            （``horse_number_a < horse_number_b``）

    Raises:
        ValueError: 対象スナップショットの列が1つも存在しない場合
    """
    _validate_snapshot(snapshot)
    path = Path(path)
    df = _read_odds_csv(path)

    pair_map: Dict[str, tuple[int, int]] = {}
    for col in df.columns:
        m = _RE_UMAREN.match(col)
        if not m or m.group(1) != snapshot:
            continue
        a, b = int(m.group(2)), int(m.group(3))
        if a > MAX_HORSE_NUMBER or b > MAX_HORSE_NUMBER:
            continue
        pair_map[col] = (min(a, b), max(a, b))
    if not pair_map:
        raise ValueError(f"馬連オッズ列（snapshot={snapshot}）が見つかりません: {path}")

    df = _dedupe_by_race_id(df, list(pair_map.keys()), path)

    sub = df[['race_id'] + list(pair_map.keys())]
    long_df = sub.melt(id_vars='race_id', var_name='_col', value_name='odds_pre_umaren')
    long_df = long_df.dropna(subset=['odds_pre_umaren'])
    pairs = long_df['_col'].map(pair_map)
    long_df['horse_number_a'] = pairs.map(lambda t: t[0]).astype('int64')
    long_df['horse_number_b'] = pairs.map(lambda t: t[1]).astype('int64')
    long_df = long_df.drop(columns='_col')
    long_df['odds_pre_umaren'] = long_df['odds_pre_umaren'].astype(float)
    out = long_df[
        ['race_id', 'horse_number_a', 'horse_number_b', 'odds_pre_umaren']
    ].reset_index(drop=True)
    logger.info(
        f"馬連締切前オッズ読込: {len(out)} 行 / {out['race_id'].nunique()} レース "
        f"(snapshot={snapshot})"
    )
    return out


def load_payouts(payouts_dir: Path | str) -> pd.DataFrame:
    """確定払戻 CSV をすべて連結して読み込む。

    ``{YYYYMM}_payouts.csv`` を対象とする。

    Args:
        payouts_dir: ``data/processed/payouts`` のパス

    Returns:
        pd.DataFrame: 全期間の払戻 DataFrame

    Raises:
        FileNotFoundError: 払戻ファイルが1つも存在しない場合
    """
    payouts_dir = Path(payouts_dir)
    files = sorted(payouts_dir.glob('*_payouts.csv'))
    if not files:
        raise FileNotFoundError(f"払戻ファイルが見つかりません: {payouts_dir}")

    frames = [pd.read_csv(f, encoding=_ENCODING, low_memory=False) for f in files]
    payouts = pd.concat(frames, ignore_index=True)
    payouts = payouts.drop_duplicates(subset='race_id', keep='last')
    logger.info(f"確定払戻読込: {len(payouts)} レース（{len(files)} ファイル）")
    return payouts


def build_win_payout(payouts: pd.DataFrame) -> pd.DataFrame:
    """確定単勝払戻を [race_id, horse_number, payout_win] に展開する。

    的中馬のみの行を返す。非的中馬は結合後に 0 で埋める。

    Args:
        payouts: ``load_payouts`` の戻り値

    Returns:
        pd.DataFrame: [race_id, horse_number, payout_win]
    """
    sub = payouts[['race_id', 'tan', 'tan_amount']].dropna(subset=['tan', 'tan_amount'])
    out = sub.rename(columns={'tan': 'horse_number', 'tan_amount': 'payout_win'})
    out['horse_number'] = out['horse_number'].astype('int64')
    out['payout_win'] = out['payout_win'].astype(float)
    # 同着1着（単勝が複数）に備え race_id × horse_number で一意化する
    return out.drop_duplicates(subset=['race_id', 'horse_number']).reset_index(drop=True)


def build_place_payout(payouts: pd.DataFrame) -> pd.DataFrame:
    """確定複勝払戻を [race_id, horse_number, payout_place] に展開する。

    同着で ``fuku_4`` まで出るため、存在する ``fuku_N`` 列を全て縦に積む。

    Args:
        payouts: ``load_payouts`` の戻り値

    Returns:
        pd.DataFrame: [race_id, horse_number, payout_place]
    """
    frames: List[pd.DataFrame] = []
    for n in range(1, 5):
        num_col, amt_col = f'fuku_{n}', f'fuku_{n}_amount'
        if num_col not in payouts.columns or amt_col not in payouts.columns:
            continue
        sub = payouts[['race_id', num_col, amt_col]].dropna(subset=[num_col, amt_col])
        frames.append(
            sub.rename(columns={num_col: 'horse_number', amt_col: 'payout_place'})
        )
    if not frames:
        raise ValueError("複勝払戻列（fuku_N / fuku_N_amount）が払戻データに存在しません")

    out = pd.concat(frames, ignore_index=True)
    out['horse_number'] = out['horse_number'].astype('int64')
    out['payout_place'] = out['payout_place'].astype(float)
    return out.drop_duplicates(subset=['race_id', 'horse_number']).reset_index(drop=True)


def build_umaren_payout(payouts: pd.DataFrame) -> pd.DataFrame:
    """確定馬連払戻を [race_id, horse_number_a, horse_number_b, payout_umaren] に展開する。

    馬番は ``a < b`` に正規化する。

    Args:
        payouts: ``load_payouts`` の戻り値

    Returns:
        pd.DataFrame: [race_id, horse_number_a, horse_number_b, payout_umaren]
    """
    cols = ['race_id', 'umaren_a', 'umaren_b', 'umaren_amount']
    missing = [c for c in cols if c not in payouts.columns]
    if missing:
        raise ValueError(f"馬連払戻列が払戻データに存在しません: {missing}")

    sub = payouts[cols].dropna(subset=['umaren_a', 'umaren_b', 'umaren_amount']).copy()
    a = sub['umaren_a'].astype('int64').to_numpy()
    b = sub['umaren_b'].astype('int64').to_numpy()
    out = pd.DataFrame({
        'race_id': sub['race_id'].to_numpy(),
        'horse_number_a': np.minimum(a, b),
        'horse_number_b': np.maximum(a, b),
        'payout_umaren': sub['umaren_amount'].astype(float).to_numpy(),
    })
    return out.drop_duplicates(
        subset=['race_id', 'horse_number_a', 'horse_number_b']
    ).reset_index(drop=True)


def build_sanfuku_payout(payouts: pd.DataFrame) -> pd.DataFrame:
    """確定三連複払戻を [race_id, horse_number_a/b/c, payout_sanfuku] に展開する。

    三連複は着順を問わない組み合わせ馬券なので、馬番を ``a < b < c`` に
    正規化してキーにする（``build_umaren_payout`` の3頭版）。

    IMPORTANT:
        戻り値の ``payout_sanfuku`` は**確定払戻＝レース結果**である。
        賭け判断に使うとリークになるため、的中後の払戻計算にのみ使うこと。
        三連複の締切前オッズは実データが存在しないため、賭け判断側は
        較正済み確率から推定した合成オッズを使う（BET_ODDS_COLS の NOTE 参照）。

    Args:
        payouts: ``load_payouts`` の戻り値

    Returns:
        pd.DataFrame: [race_id, horse_number_a, horse_number_b, horse_number_c,
            payout_sanfuku]

    Raises:
        ValueError: 三連複払戻列が存在しない場合
    """
    cols = ['race_id', 'sanfuku_a', 'sanfuku_b', 'sanfuku_c', 'sanfuku_amount']
    missing = [c for c in cols if c not in payouts.columns]
    if missing:
        raise ValueError(f"三連複払戻列が払戻データに存在しません: {missing}")

    sub = payouts[cols].dropna(
        subset=['sanfuku_a', 'sanfuku_b', 'sanfuku_c', 'sanfuku_amount']
    ).copy()
    # 3頭の馬番を昇順に正規化する（(3,1,2) と (1,2,3) を同一視する）
    nums = np.sort(
        sub[['sanfuku_a', 'sanfuku_b', 'sanfuku_c']].astype('int64').to_numpy(), axis=1
    )
    out = pd.DataFrame({
        'race_id': sub['race_id'].to_numpy(),
        'horse_number_a': nums[:, 0],
        'horse_number_b': nums[:, 1],
        'horse_number_c': nums[:, 2],
        'payout_sanfuku': sub['sanfuku_amount'].astype(float).to_numpy(),
    })
    return out.drop_duplicates(
        subset=['race_id', 'horse_number_a', 'horse_number_b', 'horse_number_c']
    ).reset_index(drop=True)


def _log_coverage(df: pd.DataFrame, col: str, label: str) -> None:
    """メタ列のカバレッジ（有効行数と対象期間）をログ出力する。

    欠損を補完で握り潰さないため、どれだけの行が評価対象外になるかを明示する。

    Args:
        df: メタ列を結合済みの DataFrame
        col: カバレッジを測る列名
        label: ログ表示用のラベル
    """
    valid = df[col].notna()
    n_valid = int(valid.sum())
    ratio = (n_valid / len(df) * 100) if len(df) > 0 else 0.0
    msg = f"[{label}] 有効行: {n_valid}/{len(df)} ({ratio:.1f}%)"
    if n_valid > 0 and 'date' in df.columns:
        dates = df.loc[valid, 'date']
        msg += f" 期間: {pd.to_datetime(dates).min().date()} 〜 {pd.to_datetime(dates).max().date()}"
    logger.info(msg)


def attach_meta_columns(
    df: pd.DataFrame,
    config: dict,
    bet_types: Iterable[str] = ('win', 'place'),
    snapshot: Optional[str] = None,
) -> pd.DataFrame:
    """馬単位 DataFrame に回収率評価用のメタ列を結合する。

    賭け判断用の締切前オッズ（``odds_pre_win`` / ``odds_pre_place``）と、
    払戻用の確定払戻金（``payout_win`` / ``payout_place``）を付与する。
    いずれも**評価専用のメタ列**であり、特徴量には含めない。

    複勝の代表オッズ ``odds_pre_place`` は ``evaluation.place_odds_mode``
    （'low'（既定） / 'mid' / 'high'）で下限・中央・上限を切り替える。
    期待値の系統的な過大評価を避けるため既定は下限。

    Args:
        df: 馬単位の DataFrame（``race_id`` / ``horse_number`` 列必須）
        config: 全体設定辞書（``data`` / ``evaluation`` セクションを参照）
        bet_types: メタ列を付与する馬券種（'win' / 'place'）
        snapshot: 締切前スナップショット時点（'10m' / '5m' / '1m'）。
            省略時は ``config['evaluation']['odds_snapshot']``（既定 '10m'）を使う。

    Returns:
        pd.DataFrame: メタ列を結合した DataFrame（入力は破壊しない）

    Raises:
        ValueError: 必須キー列が無い場合、または全行で締切前オッズが欠損する場合
    """
    for key in ('race_id', 'horse_number'):
        if key not in df.columns:
            raise ValueError(f"メタ列の結合に必要な列がありません: {key}")

    data_cfg = config.get('data', {})
    odds_dir = Path(data_cfg.get('odds_series_dir', 'data/processed/odds'))
    payouts_dir = Path(data_cfg.get('payouts_dir', 'data/processed/payouts'))
    eval_cfg = config.get('evaluation', {})
    place_odds_mode = eval_cfg.get('place_odds_mode', 'low')
    if snapshot is None:
        snapshot = eval_cfg.get('odds_snapshot', '10m')

    out = df.copy()
    payouts = load_payouts(payouts_dir)
    bet_types = tuple(bet_types)

    if 'win' in bet_types:
        pre = load_tansho_odds(odds_dir / FILENAMES['tansho'], snapshot)
        out = out.merge(pre, on=['race_id', 'horse_number'], how='left')
        pay = build_win_payout(payouts)
        out = out.merge(pay, on=['race_id', 'horse_number'], how='left')
        out['payout_win'] = out['payout_win'].fillna(0.0)
        _log_coverage(out, 'odds_pre_win', f'単勝 締切前オッズ({snapshot})')
        if out['odds_pre_win'].notna().sum() == 0:
            raise ValueError(
                "単勝の締切前オッズが全行で欠損しています。"
                f"{odds_dir / FILENAMES['tansho']} と特徴量の race_id / horse_number を確認してください。"
            )

    if 'place' in bet_types:
        pre = load_fukusho_odds(odds_dir / FILENAMES['fukusho'], snapshot)
        out = out.merge(pre, on=['race_id', 'horse_number'], how='left')
        out['odds_pre_place'] = _select_place_odds(out, place_odds_mode)
        pay = build_place_payout(payouts)
        out = out.merge(pay, on=['race_id', 'horse_number'], how='left')
        out['payout_place'] = out['payout_place'].fillna(0.0)
        _log_coverage(out, 'odds_pre_place', f'複勝 締切前オッズ({snapshot}/{place_odds_mode})')
        if out['odds_pre_place'].notna().sum() == 0:
            raise ValueError(
                "複勝の締切前オッズが全行で欠損しています。"
                f"{odds_dir / FILENAMES['fukusho']} と特徴量の race_id / horse_number を確認してください。"
            )

    return out


def _select_place_odds(df: pd.DataFrame, mode: str) -> pd.Series:
    """複勝オッズのレンジから代表値を選ぶ。

    Args:
        df: ``odds_pre_place_low`` / ``odds_pre_place_high`` を持つ DataFrame
        mode: 'low'（下限・既定） / 'mid'（中央） / 'high'（上限）

    Returns:
        pd.Series: 代表複勝オッズ

    Raises:
        ValueError: サポート外の mode が指定された場合
    """
    low = df['odds_pre_place_low']
    high = df['odds_pre_place_high']
    if mode == 'low':
        return low
    if mode == 'high':
        return high
    if mode == 'mid':
        return (low + high) / 2.0
    raise ValueError(
        f"サポートされていない place_odds_mode: {mode}（'low' / 'mid' / 'high' のいずれか）"
    )


def attach_umaren_meta_columns(
    pairs: pd.DataFrame,
    config: dict,
    snapshot: Optional[str] = None,
) -> pd.DataFrame:
    """馬連ペア DataFrame に締切前オッズと確定払戻を結合する。

    Args:
        pairs: ペア単位 DataFrame（``race_id`` / ``horse_number_a`` / ``horse_number_b`` 必須）
        config: 全体設定辞書
        snapshot: 締切前スナップショット時点（'10m' / '5m' / '1m'）。
            省略時は ``config['evaluation']['odds_snapshot']``（既定 '10m'）を使う。

    Returns:
        pd.DataFrame: ``odds_pre_umaren`` / ``payout_umaren`` を結合した DataFrame

    Raises:
        ValueError: 必須キー列が無い場合、または全行で締切前オッズが欠損する場合
    """
    keys = ['race_id', 'horse_number_a', 'horse_number_b']
    for key in keys:
        if key not in pairs.columns:
            raise ValueError(f"馬連メタ列の結合に必要な列がありません: {key}")

    data_cfg = config.get('data', {})
    odds_dir = Path(data_cfg.get('odds_series_dir', 'data/processed/odds'))
    payouts_dir = Path(data_cfg.get('payouts_dir', 'data/processed/payouts'))
    if snapshot is None:
        snapshot = config.get('evaluation', {}).get('odds_snapshot', '10m')

    out = pairs.copy()
    pre = load_umaren_odds(odds_dir / FILENAMES['umaren'], snapshot)
    out = out.merge(pre, on=keys, how='left')

    payouts = load_payouts(payouts_dir)
    pay = build_umaren_payout(payouts)
    out = out.merge(pay, on=keys, how='left')
    out['payout_umaren'] = out['payout_umaren'].fillna(0.0)

    _log_coverage(out, 'odds_pre_umaren', f'馬連 締切前オッズ({snapshot})')
    if out['odds_pre_umaren'].notna().sum() == 0:
        raise ValueError(
            "馬連の締切前オッズが全行で欠損しています。"
            f"{odds_dir / FILENAMES['umaren']} とペアの race_id / 馬番を確認してください。"
        )
    return out


# 回収率評価に使う「賭け判断オッズ」列名（馬券種 → 列名）
#
# NOTE (sanfuku): 三連複には締切前オッズの実データが存在しない
# （data/processed/odds に tansho / fukusho / umaren の3ファイルしかない）。
# ``odds_pre_sanfuku`` は較正済み確率と控除率から**推定した合成オッズ**であり、
# 市場の実オッズではない。確定払戻から逆算すると賭け判断に結果が混入して
# リークになるため、推定値で代用している。詳細は build_sanfuku_payout の
# docstring および tests/recovery_rate_backtest.ipynb の三連複セクションを参照。
BET_ODDS_COLS: Dict[str, str] = {
    'win': 'odds_pre_win',
    'place': 'odds_pre_place',
    'umaren': 'odds_pre_umaren',
    'sanfuku': 'odds_pre_sanfuku',
}

# 回収率評価に使う「確定払戻」列名（馬券種 → 列名、100円あたり）
PAYOUT_COLS: Dict[str, str] = {
    'win': 'payout_win',
    'place': 'payout_place',
    'umaren': 'payout_umaren',
    'sanfuku': 'payout_sanfuku',
}


def has_two_source_odds(df: pd.DataFrame, bet_type: str) -> bool:
    """2系統オッズ（締切前オッズ＋確定払戻）が揃っているかを判定する。

    Args:
        df: 判定対象の DataFrame
        bet_type: 'win' / 'place' / 'umaren'

    Returns:
        bool: 両方の列が存在すれば True
    """
    bet_col = BET_ODDS_COLS.get(bet_type)
    payout_col = PAYOUT_COLS.get(bet_type)
    if bet_col is None or payout_col is None:
        return False
    return bet_col in df.columns and payout_col in df.columns


def meta_columns(bet_types: Optional[Iterable[str]] = None) -> List[str]:
    """評価専用メタ列の一覧を返す（features.json の exclude_cols 検査用）。

    Args:
        bet_types: 対象馬券種。省略時は全馬券種。

    Returns:
        List[str]: メタ列名のリスト
    """
    types = tuple(bet_types) if bet_types is not None else tuple(BET_ODDS_COLS.keys())
    cols: List[str] = []
    for t in types:
        cols.append(BET_ODDS_COLS[t])
        cols.append(PAYOUT_COLS[t])
    cols.extend(['odds_pre_place_low', 'odds_pre_place_high'])
    return cols
