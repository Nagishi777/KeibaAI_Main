"""当日オッズ・投票数による予測とモデル再学習のパッケージ。

``docs/pool_filter_condition_spec.md`` の条件（pool q0.75 / all / ev1.10）を、
判断時点を 5m に前倒し（賭け5m / pool_5m / 特徴量 odds_5m・inc_share_10m_5m）
して実運用の2プログラムに落としたもの。

    # レース当日: 5m時点オッズを取得して予測・ベット推奨を出す
    python -m src.simulator.predict_today

    # レース後: 確定着順を加えてモデルを再学習する
    python -m src.simulator.retrain --start-period 202401 --end-period 202607

双方が :mod:`src.simulator.features` の同一関数で特徴量を組むため、
学習時と推論時で特徴量の定義がずれることはない。

IMPORTANT:
    本条件は単一分割・多重比較（108条件の探索）を含む調査結果であり、
    walk-forward による期間安定性は未検証である。実運用の推奨ではない。
    限界は ``docs/pool_filter_condition_spec.md`` §8 を必ず読むこと。
"""
