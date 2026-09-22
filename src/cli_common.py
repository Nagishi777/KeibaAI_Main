"""
CLI共通ユーティリティ

src/*/*.py の単独実行エントリポイントから共通利用する
設定ロード・ロギング設定をまとめたモジュール。
"""
import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Sequence, Union

import yaml


class ModelFeatureSchemaError(ValueError):
    """モデル特徴量スキーマの検証エラー。"""


def model_feature_path(config: dict) -> Path:
    """model_dir 配下のモデル特徴量スキーマJSONのパスを返す。"""
    return Path(config['data'].get('model_dir', 'data/model')) / 'model_feature.json'


def derive_model_feature_cols(columns: Sequence[str], features_def: dict) -> list[str]:
    """学習・推論共通のルールでモデル入力特徴量を抽出する。"""
    non_feature_cols = set(features_def.get('model_keep_cols', []))
    return [col for col in columns if col not in non_feature_cols]


def derive_expert_feature_cols(
    columns: Sequence[str], features_def: dict, group: str
) -> list[str]:
    """スタッキングの1専門家（base learner）の入力特徴量を解決する。

    ``derive_model_feature_cols`` の結果（＝モノリスの全特徴量）を
    **部分集合化**する形で定義する。並列な別系統にすると
    「モデル入力列の真実の源」が2つになるため。

    ``config/features.json`` の ``feature_groups.<group>`` を参照し、
    以下のキーを順に適用する。

    - ``include_prefixes``: いずれかのプレフィックスで始まる列を採用
    - ``include_explicit``: 列名を直接指定して採用
    - ``include_derived_suffixes``: 採用済み列の派生列
      （``_race_zscore`` / ``_race_rank_pct`` 等）も併せて採用
    - ``exclude_prefixes``: いずれかのプレフィックスで始まる列を除外

    ``include_*`` がひとつも無い場合は全特徴量を起点とし、
    ``exclude_prefixes`` だけを適用する（E1 能力モデルの想定）。

    Args:
        columns: 特徴量ファイルの全列名
        features_def: ``config/features.json`` の内容
        group: ``feature_groups`` のキー名

    Returns:
        list[str]: 専門家の入力列（入力の並び順を保持）

    Raises:
        ModelFeatureSchemaError: グループ定義が存在しない場合、
            または解決結果が0列になった場合（想定外は停止する）
    """
    groups = features_def.get('feature_groups', {})
    if group not in groups:
        raise ModelFeatureSchemaError(
            f"未知の特徴量グループです: '{group}'. "
            f"config/features.json の feature_groups に定義してください"
            f"（定義済み: {sorted(groups.keys())}）"
        )
    spec = groups[group]
    base_cols = derive_model_feature_cols(columns, features_def)

    include_prefixes = tuple(spec.get('include_prefixes', []))
    include_explicit = set(spec.get('include_explicit', []))

    if include_prefixes or include_explicit:
        selected = [
            c for c in base_cols
            if (include_prefixes and c.startswith(include_prefixes))
            or c in include_explicit
        ]
        # 採用済み列の派生列（レース内相対）も取り込む
        suffixes = tuple(spec.get('include_derived_suffixes', []))
        if suffixes:
            chosen = set(selected)
            derived = [
                c for c in base_cols
                if c not in chosen
                and c.endswith(suffixes)
                and any(c.startswith(p) for p in include_prefixes)
            ]
            selected = [c for c in base_cols if c in set(selected) | set(derived)]
    else:
        selected = list(base_cols)

    exclude_prefixes = tuple(spec.get('exclude_prefixes', []))
    if exclude_prefixes:
        selected = [c for c in selected if not c.startswith(exclude_prefixes)]

    if not selected:
        raise ModelFeatureSchemaError(
            f"特徴量グループ '{group}' の解決結果が0列です。"
            f"config/features.json の定義と特徴量ファイルの列名を確認してください。"
        )
    return selected


def expert_feature_path(config: dict, expert_id: str) -> Path:
    """専門家ごとの特徴量スキーマJSONのパスを返す。"""
    model_dir = Path(config['data'].get('model_dir', 'data/model'))
    return model_dir / 'stack' / f'{expert_id}_feature.json'


def save_expert_feature_schema(
    config: dict, expert_id: str, feature_cols: Iterable[str]
) -> Path:
    """専門家ごとの特徴量スキーマを保存する。

    ``model_feature.json`` と同じ ``{"model_feature_cols": [...]}`` 形式にして、
    ``validate_model_feature_columns_strict`` をそのまま再利用できるようにする。
    """
    path = expert_feature_path(config, expert_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_feature_cols': list(feature_cols)}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_expert_feature_schema(config: dict, expert_id: str) -> tuple[list[str], Path]:
    """専門家ごとの特徴量スキーマを読み込む。

    Raises:
        FileNotFoundError: スキーマが存在しない場合
        ModelFeatureSchemaError: 形式が不正な場合
    """
    path = expert_feature_path(config, expert_id)
    if not path.exists():
        raise FileNotFoundError(
            f"専門家 '{expert_id}' の特徴量スキーマが見つかりません: {path}. "
            "先にスタック学習を実行してください"
        )
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    cols = payload.get('model_feature_cols')
    if not isinstance(cols, list) or not cols:
        raise ModelFeatureSchemaError(
            f"スキーマ形式が不正です: {path}（model_feature_cols が空、または存在しません）"
        )
    return [str(c) for c in cols], path


def save_model_feature_schema(config: dict, feature_cols: Iterable[str]) -> Path:
    """モデル特徴量スキーマを model_dir に保存する。"""
    path = model_feature_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'model_feature_cols': list(feature_cols)}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def load_model_feature_schema(config: dict) -> tuple[list[str], Path]:
    """model_dir からモデル特徴量スキーマを読み込む。"""
    path = model_feature_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"モデル特徴量スキーマが見つかりません: {path}. "
            "先に train モードを実行してください"
        )

    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    cols = payload.get('model_feature_cols')
    if cols is None:
        # 後方互換（旧キー）
        cols = payload.get('feature_cols')
    if not isinstance(cols, list) or not cols:
        raise ModelFeatureSchemaError(
            f"スキーマ形式が不正です: {path}（model_feature_cols が空、または存在しません）"
        )

    return [str(c) for c in cols], path


def validate_model_feature_columns_strict(
    actual_cols: Sequence[str],
    expected_cols: Sequence[str],
    *,
    stage: str,
    schema_path: Path,
    race_id: str | None = None,
) -> None:
    """モデル入力列を厳格検証する（不足/余剰が1つでもあればエラー）。"""
    actual_set = set(actual_cols)
    expected_set = set(expected_cols)

    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if not missing and not extra:
        return

    details = [
        "モデル特徴量スキーマ不一致を検知したため処理を停止しました。",
        f"stage={stage}",
        f"schema={schema_path}",
    ]
    if race_id is not None:
        details.append(f"race_id={race_id}")
    if missing:
        details.append(f"missing_cols({len(missing)}): {missing}")
    if extra:
        details.append(f"extra_cols({len(extra)}): {extra}")

    raise ModelFeatureSchemaError('\n'.join(details))


def setup_logging(config: dict) -> None:
    """
    ログ設定

    Args:
        config: 設定辞書
    """
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO')
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = log_config.get('file', 'app.log')

    # ログディレクトリ作成
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ログ設定（ファイルは約1000行≒100KBで自動ローテーション、バックアップなし）
    # maxBytesを超えると古いログを削除して新しいログを書き込む（FIFO）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=100_000, backupCount=0, encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))

    # ターミナルには WARNING 以上のみ表示（INFO は多すぎるため）
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(logging.Formatter(log_format))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            file_handler,
            stream_handler,
        ]
    )


def load_config(config_path: Union[str, Path] = 'config/config.yaml') -> dict:
    """
    設定ファイル(config.yaml)を読み込み

    Args:
        config_path: 設定ファイルパス

    Returns:
        dict: 設定辞書
    """
    config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_features_config(features_path: Union[str, Path] = 'config/features.json') -> dict:
    """
    特徴量定義ファイル(features.json)を読み込み

    Args:
        features_path: features設定パス

    Returns:
        dict: features設定辞書
    """
    features_path = Path(features_path)
    with open(features_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_config(config_path: Union[str, Path] = 'config/config.yaml') -> dict:
    """
    config.yaml + features.json をロードし、ロギングを設定した設定辞書を返す。

    config_path と同じディレクトリの features.json を特徴量定義として読み込み、
    config['features_def'] にセットする。各モジュールの単独実行エントリポイントから
    共通利用する。

    Args:
        config_path: config.yaml のパス

    Returns:
        dict: features_def をセットした設定辞書
    """
    config_path = Path(config_path)
    config = load_config(config_path)
    config['features_def'] = load_features_config(config_path.parent / 'features.json')
    setup_logging(config)
    return config
