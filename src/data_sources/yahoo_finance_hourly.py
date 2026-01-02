"""
Yahoo Finance 時間足データ収集
すべての上下移動を捉えるための高頻度データ - 無料・無制限
"""
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import yfinance as yf
from typing import List, Optional


class YahooFinanceHourly:
    """Yahoo Finance 時間足データ収集"""

    def __init__(self, interval: str = '1h'):
        """
        Args:
            interval: 時間足の粒度
                - 1h: 1時間足（最高頻度）
                - 4h: 4時間足（中頻度）（Note: yfinanceは4hをサポートしていない可能性）
                - 1d: 日足（現在使用中）
        """
        self.interval = interval
        self.data_dir = Path('data/hourly')
        self.data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Yahoo Finance 時間足データ収集 初期化: {interval}")

    def get_hourly_data(
        self,
        pair: str = 'USDJPY=X',
        period: str = '2y',  # 2年分
        save: bool = True
    ) -> pd.DataFrame:
        """
        Yahoo Financeから時間足データを取得

        Args:
            pair: 通貨ペア（Yahoo Finance形式）
                - USDJPY=X: USD/JPY
                - EURUSD=X: EUR/USD
                - GBPUSD=X: GBP/USD
                - EURJPY=X: EUR/JPY
            period: 取得期間
                - '1mo': 1ヶ月
                - '3mo': 3ヶ月
                - '1y': 1年
                - '2y': 2年
                - 'max': 最大
            save: CSVに保存するか

        Returns:
            時間足データのDataFrame
        """
        logger.info(f"{pair} {self.interval}データ取得開始: {period}")

        try:
            # Yahoo Financeからデータ取得
            ticker = yf.Ticker(pair)
            df = ticker.history(
                period=period,
                interval=self.interval
            )

            if df.empty:
                raise ValueError(f"データを取得できませんでした: {pair}")

            # カラム名を標準化（小文字）
            df.columns = df.columns.str.lower()

            # 不要な列を削除（dividends, stock splitsはFXには不要）
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"✅ 取得完了: {len(df)}本のローソク足")
            logger.info(f"期間: {df.index.min()} ~ {df.index.max()}")
            logger.info(f"データ完全性: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")

            # 保存
            if save:
                filename = f"{pair.replace('=X', '')}_{self.interval}_{period}.csv"
                filepath = self.data_dir / filename
                df.to_csv(filepath)
                logger.info(f"💾 保存: {filepath}")

            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            raise

    def get_multi_currency_data(
        self,
        pairs: List[str] = ['USDJPY=X', 'EURUSD=X', 'GBPUSD=X', 'EURJPY=X'],
        period: str = '2y'
    ) -> dict:
        """
        複数通貨ペアのデータを一括取得

        Args:
            pairs: 通貨ペアのリスト（Yahoo Finance形式）
            period: 取得期間

        Returns:
            {通貨ペア: DataFrame} の辞書
        """
        logger.info(f"複数通貨ペアデータ取得開始: {len(pairs)}ペア")

        results = {}
        for pair in pairs:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"通貨ペア: {pair}")
                logger.info(f"{'='*60}")

                df = self.get_hourly_data(
                    pair=pair,
                    period=period,
                    save=True
                )
                results[pair] = df

            except Exception as e:
                logger.error(f"{pair} のデータ取得に失敗: {e}")
                continue

        logger.info(f"\n✅ 複数通貨ペアデータ取得完了: {len(results)}/{len(pairs)}ペア")
        return results

    def load_saved_data(self, pair: str, period: str = '2y') -> Optional[pd.DataFrame]:
        """保存済みデータを読み込み"""
        filename = f"{pair.replace('=X', '')}_{self.interval}_{period}.csv"
        filepath = self.data_dir / filename

        if filepath.exists():
            logger.info(f"保存済みデータ読み込み: {filepath}")
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            logger.info(f"✅ 読み込み完了: {len(df)}本")
            return df
        else:
            logger.warning(f"保存済みデータが見つかりません: {filepath}")
            return None


def collect_all_hourly_data():
    """すべての通貨ペアの時間足データを収集（メイン実行）"""
    logger.info("="*80)
    logger.info("世界最強システム - Yahoo Finance 時間足データ収集開始")
    logger.info("="*80)

    # 1時間足で収集
    collector = YahooFinanceHourly(interval='1h')

    # 対象通貨ペア（Yahoo Finance形式）
    pairs = [
        'USDJPY=X',   # USD/JPY - 日本時間に強い
        'EURUSD=X',   # EUR/USD - 欧州・米国時間に強い
        'GBPUSD=X',   # GBP/USD - ボラティリティ高い
        'EURJPY=X',   # EUR/JPY - クロス円の代表
    ]

    # 2年分のデータ取得
    results = collector.get_multi_currency_data(
        pairs=pairs,
        period='2y'
    )

    # サマリー表示
    logger.info("\n" + "="*80)
    logger.info("データ収集サマリー")
    logger.info("="*80)

    total_candles = 0
    for pair, df in results.items():
        logger.info(f"\n{pair}:")
        logger.info(f"  ローソク足数: {len(df):,}")
        logger.info(f"  期間: {df.index.min()} ~ {df.index.max()}")
        logger.info(f"  完全性: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
        logger.info(f"  データ範囲:")
        logger.info(f"    始値: {df['open'].min():.3f} ~ {df['open'].max():.3f}")
        logger.info(f"    終値: {df['close'].min():.3f} ~ {df['close'].max():.3f}")
        total_candles += len(df)

    logger.info(f"\n{'='*80}")
    logger.info(f"合計ローソク足数: {total_candles:,}本")
    logger.info(f"取引機会（推定）: 日次の{total_candles / (2*365):.0f}倍")
    logger.info("✅ すべてのデータ収集完了！")
    logger.info("="*80)

    return results


if __name__ == '__main__':
    collect_all_hourly_data()
