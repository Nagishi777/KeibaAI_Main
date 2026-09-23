"""学習済みモデルと運用パラメータの保存・読込モジュール。

``ModelCreator``（=``LightGBMTrainer``）は Booster と
``<stem>_calibrator.pkl`` を対で永続化する。本モジュールはそれに加えて、
予測側が必要とする**運用パラメータ**を ``<stem>_simulator.json`` に保存する。

保存するもの:

============================  ==================================================
``pool_threshold``            レース選別のしきい値（円）。学習期間の q0.75
``pool_quantile``             しきい値を取った分位点
``feature_cols``              モデル入力の列と順序
``min_ev`` / ``odds_snapshot``  賭け条件（予測側の既定値）
``train_start`` / ``train_end`` 学習期間（しきい値の出どころを追えるように）
============================  ==================================================

IMPORTANT (しきい値を保存する理由):
    ``pool_threshold`` は**学習期間の分布**から決め、その固定値を将来へ
    適用する（spec §5.2）。予測時に当日のレース群から分位点を取り直すと、
    「その日たまたま厚かったレース」を基準にしてしまい、
    未来を見ない前提が崩れる。よって学習時に確定させて持ち回る。
"""
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from src.simulator.features import DECISION_SNAPSHOT, FEATURE_COLS

logger = logging.getLogger(__name__)

# モデルファイル名（``data/model`` からの相対パス）
DEFAULT_MODEL_FILENAME: str = 'simulator/win_pool_filter.txt'

# 運用パラメータの接尾辞（Booster と同じ stem に付ける）
PARAMS_SUFFIX: str = '_simulator.json'


@dataclass
class SimulatorParams:
    """予測側が必要とする運用パラメータ。"""

    # spec §5: レース選別のしきい値（円）と、その出どころ
    pool_threshold: float
    pool_quantile: float
    # spec §3: モデル入力の列と順序
    feature_cols: List[str] = field(default_factory=lambda: list(FEATURE_COLS))
    # spec §6: 賭け条件
    min_ev: float = 1.10
    odds_snapshot: str = DECISION_SNAPSHOT
    # 学習期間（しきい値の根拠を追えるようにする）
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
        FileNotFoundError: JSON が存在しない場合（しきい値が不明なまま
            予測すると全レースを賭け対象にしてしまうため停止する）
        ValueError: 特徴量カラムが現在の定義と食い違う場合
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

    params = SimulatorParams(**raw)
    if list(params.feature_cols) != list(FEATURE_COLS):
        raise ValueError(
            'モデルの特徴量カラムが現在の定義と一致しません。\n'
            f'  モデル: {params.feature_cols}\n'
            f'  現在:   {list(FEATURE_COLS)}\n'
            ' src.simulator.features を変更した場合は再学習が必要です。'
        )
    logger.info(
        'モデル運用パラメータ: pool_threshold=%s円（q%.2f / 学習 %s〜%s）'
        ' / min_ev=%.2f / snapshot=%s',
        f'{params.pool_threshold:,.0f}', params.pool_quantile,
        params.train_start, params.train_end, params.min_ev,
        params.odds_snapshot,
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
    EV 閾値・Kelly 計算が静かに壊れるため、欠けていたら停止する。

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
            ' 校正前の確率で EV・Kelly を計算すると賭け金が壊れるため停止します。'
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
