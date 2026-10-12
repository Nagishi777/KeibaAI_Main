"""
CLI共通ユーティリティ

src/*/*.py の単独実行エントリポイントから共通利用する
設定ロード・ロギング設定をまとめたモジュール。
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Union

import yaml


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


def load_config(config_path: Union[str, Path] = 'config.yaml') -> dict:
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


def build_config(config_path: Union[str, Path] = 'config.yaml') -> dict:
    """
    config.yaml をロードし、ロギングを設定した設定辞書を返す。

    各モジュールの単独実行エントリポイントから共通利用する。

    Args:
        config_path: config.yaml のパス

    Returns:
        dict: 設定辞書
    """
    config = load_config(config_path)
    setup_logging(config)
    return config
