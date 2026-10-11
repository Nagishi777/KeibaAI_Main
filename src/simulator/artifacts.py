"""学習済みモデルと運用パラメータの保存・読込モジュール。

``ModelCreator``（=``LightGBMTrainer``）は Booster と
``<stem>_calibrator.pkl`` を対で永続化する。本モジュールはそれに加えて、
予測側がモデルの前提を検証するための情報を ``<stem>_simulator.json`` に
保存する。

保存するもの:

===============================  ===============================================
``feature_cols``                 モデル入力の列と順序
``train_decision_snapshot``      学習で締切直前として使った時点（``1m``）
``drift_base_snapshot``          ドリフトの起点（``5m``）
``train_start`` / ``train_end``  学習期間
``holdout_auc`` / ``num_trees``  学習結果の参考値
===============================  ===============================================

IMPORTANT (購入条件はここに保存しない):
    EV・オッズ帯・ドリフト・総投票額のしきい値は
    :mod:`src.simulator.strategy` が持ち、``config.yaml`` の ``strategy``
    セクションで上書きする。モデルの再学習とは独立に変えられるべき値で
    あり、モデルファイルに焼き付けると「どちらが効いているのか」が
    追えなくなるため。

IMPORTANT (学習と当日で時点が違うことを検証する):
    学習は ``1m``、当日判断は ``2m`` である（過去データに 2m が無いため）。
    :func:`load_params` はこのずれをログに出す。黙って進めると、後から
    「どの時点で学習したモデルなのか」が分からなくなるため。
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.simulator.features import (
    DRIFT_BASE_SNAPSHOT,
    FEATURE_COLS,
    LIVE_DECISION_SNAPSHOT,
    TRAIN_DECISION_SNAPSHOT,
)

logger = logging.getLogger(__name__)

# モデルファイル名（``data/model`` からの相対パス）
DEFAULT_MODEL_FILENAME: str = 'win/win_anaba.txt'

# 運用パラメータの接尾辞（Booster と同じ stem に付ける）
PARAMS_SUFFIX: str = '_simulator.json'


@dataclass
class SimulatorParams:
    """予測側がモデルの前提を検証するためのパラメータ。"""

    # モデル入力の列と順序
    feature_cols: List[str] = field(default_factory=lambda: list(FEATURE_COLS))
    # 学習で締切直前として使った時点と、ドリフトの起点
    train_decision_snapshot: str = TRAIN_DECISION_SNAPSHOT
    drift_base_snapshot: str = DRIFT_BASE_SNAPSHOT
    # 学習期間（モデルの出どころを追えるようにする）
    train_start: str = ''
    train_end: str = ''
    n_train_rows: int = 0
    n_train_races: int = 0
    # 学習結果の参考値
    holdout_auc: float = float('nan')
    num_trees: int = 0

    def params_path(self, model_path: Path) -> Path:
        """Booster と対になる JSON のパスを返す。

        Args:
            model_path: Booster のパス

        Returns:
            Path: 運用パラメータ JSON のパス
        """
        return model_path.with_name(f'{model_path.stem}{PARAMS_SUFFIX}')


def save_params(params: SimulatorParams, model_path: Path) -> Path:
    """運用パラメータを Booster と同じ場所へ保存する。

    Args:
        params: 保存するパラメータ
        model_path: Booster のパス

    Returns:
        Path: 書き出した JSON のパス
    """
    path = params.params_path(Path(model_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(asdict(params), f, ensure_ascii=False, indent=2)
    logger.info('運用パラメータを保存: %s', path)
    return path


def load_params(model_path: Path) -> SimulatorParams:
    """Booster と対になる運用パラメータを読み込む。

    Args:
        model_path: Booster のパス

    Returns:
        SimulatorParams: 読み込んだパラメータ

    Raises:
        FileNotFoundError: JSON が存在しない場合
        ValueError: 特徴量カラムが現在の定義と食い違う場合（列順が違うと
            LightGBM の予測が黙って壊れるため停止する）
    """
    model_path = Path(model_path)
    path = model_path.with_name(f'{model_path.stem}{PARAMS_SUFFIX}')
    if not path.exists():
        raise FileNotFoundError(
            f'運用パラメータが見つかりません: {path}。'
            ' 先に python -m src.simulator.retrain を実行してください。'
        )
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    known = {f.name for f in SimulatorParams.__dataclass_fields__.values()}
    unknown = sorted(set(raw) - known)
    if unknown:
        # 旧形式（pool_threshold / min_ev を持つ）を読んだ場合はここで落ちる。
        # 特徴量定義も変わっているため、再学習が必要である。
        raise ValueError(
            f'運用パラメータ {path} に未知の項目があります: {unknown}。'
            ' 戦略の変更前に保存されたモデルの可能性があります。'
            ' python -m src.simulator.retrain で再学習してください。'
        )

    params = SimulatorParams(**raw)
    if list(params.feature_cols) != list(FEATURE_COLS):
        raise ValueError(
            'モデルの特徴量カラムが現在の定義と一致しません。\n'
            f'  モデル: {params.feature_cols}\n'
            f'  現在:   {list(FEATURE_COLS)}\n'
            ' src.simulator.features を変更した場合は再学習が必要です。'
        )
    logger.info(
        'モデル運用パラメータ: 特徴量 %d 列 / 学習 %s〜%s（%s レース）'
        ' / holdout AUC %.4f / 学習時点 %s',
        len(params.feature_cols), params.train_start, params.train_end,
        f'{params.n_train_races:,}', params.holdout_auc,
        params.train_decision_snapshot,
    )
    if params.train_decision_snapshot != LIVE_DECISION_SNAPSHOT:
        logger.warning(
            'このモデルは %s 時点で学習し、当日は %s 時点で判断します'
            '（過去データに %s が無いための既知のずれ）。',
            params.train_decision_snapshot, LIVE_DECISION_SNAPSHOT,
            LIVE_DECISION_SNAPSHOT,
        )
    return params


def resolve_model_path(model_dir: Path, filename: str) -> Path:
    """``data/model`` 配下のモデルパスを解決する。

    Args:
        model_dir: ``data/model`` のパス
        filename: モデルファイル名（model_dir からの相対パス）

    Returns:
        Path: Booster の絶対/相対パス
    """
    return Path(model_dir) / filename


def require_model(model_path: Path) -> None:
    """モデルとキャリブレータの存在を確認する。

    キャリブレータが無いまま予測すると確率のスケールが学習時とずれ、
    EV 閾値が静かに壊れる（EV = 予測確率 × オッズ は確率が校正されている
    ことを前提にしている）ため、欠けていたら停止する。

    Args:
        model_path: Booster のパス

    Raises:
        FileNotFoundError: Booster またはキャリブレータが無い場合
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f'学習済みモデルが見つかりません: {model_path}。'
            ' 先に python -m src.simulator.retrain を実行してください。'
        )
    calibrator = model_path.with_name(f'{model_path.stem}_calibrator.pkl')
    if not calibrator.exists():
        raise FileNotFoundError(
            f'キャリブレータが見つかりません: {calibrator}。'
            ' 校正前の確率で EV を計算すると購入判定が壊れるため停止します。'
            ' 再学習してください。'
        )


def latest_params(model_dir: Path, filename: str) -> Optional[SimulatorParams]:
    """保存済みパラメータがあれば読む（無ければ None）。

    Args:
        model_dir: ``data/model`` のパス
        filename: モデルファイル名

    Returns:
        Optional[SimulatorParams]: 読み込めたパラメータ
    """
    try:
        return load_params(resolve_model_path(model_dir, filename))
    except FileNotFoundError:
        return None
