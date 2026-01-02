"""
OANDA時間足データ収集
すべての上下移動を捉えるための高頻度データ取得
"""
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import time
from typing import List, Optional

from ..api.oanda_client import OandaClient


class OandaHourlyData:
    """OANDA時間足データ収集"""

    def __init__(self, granularity: str = 'H1'):
        """
        Args:
            granularity: 時間足の粒度
                - H1: 1時間足（最高頻度）
                - H4: 4時間足（中頻度）
                - D: 日足（現在使用中）
        """
        self.client = OandaClient()
        self.granularity = granularity
        self.data_dir = Path('data/hourly')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"OANDA時間足データ収集 初期化: {granularity}")

    def get_hourly_data(
        self,
        instrument: str = 'USD_JPY',
        days: int = 730,  # 2年分
        save: bool = True
    ) -> pd.DataFrame:
        """
        大量の時間足データを取得（最新のcountデータのみ）

        Note: OANDA APIの制限により、一度に取得できるのは最新500本まで。
              2年分の全データを取得するには、Yahoo Financeの方が適している可能性あり。

        Args:
            instrument: 通貨ペア
            days: 取得日数（デフォルト730日 = 2年）
            save: CSVに保存するか

        Returns:
            時間足データのDataFrame
        """
        logger.info(f"{instrument} {self.granularity}データ取得開始")

        # まず最新500本を取得
        try:
            logger.info("最新データを取得中...")
            df = self.client.get_historical_data(
                instrument=instrument,
                granularity=self.granularity,
                count=500  # 最大500本
            )

            if df.empty:
                raise ValueError("データを取得できませんでした")

            logger.info(f"✅ 取得完了: {len(df)}本のローソク足")
            logger.info(f"期間: {df.index.min()} ~ {df.index.max()}")
            logger.info(f"⚠️ OANDA APIの制限により、最新500本のみ取得")
            logger.info(f"💡 より長期データが必要な場合は、Yahoo Financeを検討してください")

            # 保存
            if save:
                filename = f"{instrument}_{self.granularity}_latest.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"💾 保存: {filepath}")

            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            raise

    def _calculate_total_candles(self, days: int) -> int:
        """必要なローソク足数を計算"""
        if self.granularity == 'H1':
            # 1時間足: 1日24本（土日除外で約20本/日）
            return days * 20
        elif self.granularity == 'H4':
            # 4時間足: 1日6本（土日除外で約5本/日）
            return days * 5
        elif self.granularity == 'D':
            # 日足: 1日1本（土日除外で約0.7本/日）
            return days
        else:
            # デフォルト
            return days * 20

    def get_multi_currency_data(
        self,
        instruments: List[str] = ['USD_JPY', 'EUR_USD', 'GBP_USD', 'EUR_JPY'],
        days: int = 730
    ) -> dict:
        """
        複数通貨ペアのデータを一括取得

        Args:
            instruments: 通貨ペアのリスト
            days: 取得日数

        Returns:
            {通貨ペア: DataFrame} の辞書
        """
        logger.info(f"複数通貨ペアデータ取得開始: {len(instruments)}ペア")

        results = {}
        for instrument in instruments:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"通貨ペア: {instrument}")
                logger.info(f"{'='*60}")

                df = self.get_hourly_data(
                    instrument=instrument,
                    days=days,
                    save=True
                )
                results[instrument] = df

                # 通貨ペア間で1秒待機（APIレート制限）
                time.sleep(1)

            except Exception as e:
                logger.error(f"{instrument} のデータ取得に失敗: {e}")
                continue

        logger.info(f"\n✅ 複数通貨ペアデータ取得完了: {len(results)}/{len(instruments)}ペア")
        return results

    def load_saved_data(self, instrument: str, days: int = 730) -> Optional[pd.DataFrame]:
        """保存済みデータを読み込み"""
        filename = f"{instrument}_{self.granularity}_{days}days.csv"
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"保存済みデータ読み込み: {filepath}")
            df = pd.read_csv(filepath, index_col='time', parse_dates=True)
            logger.info(f"✅ 読み込み完了: {len(df)}本")
            return df
        else:
            logger.warning(f"保存済みデータが見つかりません: {filepath}")
            return None


def collect_all_hourly_data():
    """すべての通貨ペアの時間足データを収集（メイン実行）"""
    logger.info("="*80)
    logger.info("世界最強システム - 時間足データ収集開始")
    logger.info("="*80)

    # H1（1時間足）で収集
    collector = OandaHourlyData(granularity='H1')

    # 対象通貨ペア
    instruments = [
        'USD_JPY',   # 日本時間に強い
        'EUR_USD',   # 欧州・米国時間に強い
        'GBP_USD',   # ボラティリティ高い
        'EUR_JPY',   # クロス円の代表
    ]

    # 2年分のデータ取得
    results = collector.get_multi_currency_data(
        instruments=instruments,
        days=730
    )

    # サマリー表示
    logger.info("\n" + "="*80)
    logger.info("データ収集サマリー")
    logger.info("="*80)
    for instrument, df in results.items():
        logger.info(f"{instrument}:")
        logger.info(f"  ローソク足数: {len(df)}")
        logger.info(f"  期間: {df.index.min()} ~ {df.index.max()}")
        logger.info(f"  完全性: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")

    logger.info("\n✅ すべてのデータ収集完了！")
    return results


if __name__ == '__main__':
    collect_all_hourly_data()
