"""
Yahoo Finance データソース

長期間の為替データを無料で取得
yfinanceライブラリ使用
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from typing import Optional
import warnings
warnings.filterwarnings('ignore')


class YahooFinanceData:
    """
    Yahoo Financeから為替データ取得

    利点:
    - 無料
    - 長期間データ（10年以上）
    - API key不要
    """

    def __init__(self):
        """初期化"""
        try:
            import yfinance as yf
            self.yf = yf
            logger.info("Yahoo Finance データソース初期化成功")
        except ImportError:
            logger.error("yfinance未インストール: pip install yfinance")
            raise

    def get_forex_data(
        self,
        pair: str = "USD/JPY",
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        為替データ取得

        Args:
            pair: 通貨ペア (USD/JPY, EUR/USD等)
            start_date: 開始日 (YYYY-MM-DD)
            end_date: 終了日 (YYYY-MM-DD、Noneなら今日)
            interval: 時間足 (1d, 1h, 1wk等)

        Returns:
            DataFrame (open, high, low, close, volume)
        """
        # Yahoo Financeのティッカー形式に変換
        ticker = self._pair_to_ticker(pair)

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"{pair} データ取得中: {start_date} ~ {end_date}")

        try:
            # データ取得
            data = self.yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False
            )

            if data.empty:
                logger.warning(f"{ticker} データが空です")
                return pd.DataFrame()

            # カラム名を小文字に統一（tupleの場合は最初の要素を取得）
            if isinstance(data.columns[0], tuple):
                data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
            else:
                data.columns = [col.lower() for col in data.columns]

            # volumeがない場合は0で埋める
            if 'volume' not in data.columns:
                data['volume'] = 0

            logger.info(f"{pair} データ取得完了: {len(data)}件")

            return data[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"{pair} データ取得エラー: {e}")
            return pd.DataFrame()

    def get_multiple_pairs(
        self,
        pairs: list,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> dict:
        """
        複数通貨ペア一括取得

        Args:
            pairs: 通貨ペアリスト
            start_date: 開始日
            end_date: 終了日

        Returns:
            {pair: DataFrame} の辞書
        """
        result = {}

        for pair in pairs:
            data = self.get_forex_data(pair, start_date, end_date)
            if not data.empty:
                result[pair] = data

            # レート制限回避
            import time
            time.sleep(0.5)

        logger.info(f"複数通貨ペア取得完了: {len(result)}ペア")
        return result

    def _pair_to_ticker(self, pair: str) -> str:
        """
        通貨ペアをYahoo Financeティッカーに変換

        Args:
            pair: USD/JPY形式

        Returns:
            Yahoo Financeティッカー (例: JPY=X)
        """
        # USD/JPY -> JPYUSD=X (Yahoo形式)
        # スラッシュを削除して=Xを追加
        pair_clean = pair.replace('/', '').replace('_', '')

        # Yahoo Financeでは逆順+Xの場合が多い
        # USD/JPY -> JPY=X ではなく USDJPY=X
        ticker = f"{pair_clean}=X"

        return ticker

    def get_stock_indices(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> dict:
        """
        主要株価指数取得

        Returns:
            {index_name: DataFrame} の辞書
        """
        indices = {
            'SP500': '^GSPC',      # S&P 500
            'NIKKEI': '^N225',     # 日経平均
            'DOW': '^DJI',         # ダウ平均
            'NASDAQ': '^IXIC',     # NASDAQ
            'VIX': '^VIX'          # 恐怖指数
        }

        result = {}

        for name, ticker in indices.items():
            try:
                data = self.yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    interval='1d',
                    progress=False
                )

                if not data.empty:
                    # カラム名を小文字に統一（tupleの場合は最初の要素を取得）
                    if isinstance(data.columns[0], tuple):
                        data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
                    else:
                        data.columns = [col.lower() for col in data.columns]

                    if 'volume' not in data.columns:
                        data['volume'] = 0
                    result[name] = data[['open', 'high', 'low', 'close', 'volume']]
                    logger.info(f"{name} ({ticker}) 取得: {len(data)}件")

            except Exception as e:
                logger.error(f"{name} 取得エラー: {e}")

            import time
            time.sleep(0.5)

        return result


def test_yahoo_finance():
    """動作テスト"""
    logger.info("Yahoo Finance データソーステスト開始")

    yf_data = YahooFinanceData()

    # USD/JPY 取得（3年分）
    usd_jpy = yf_data.get_forex_data(
        pair="USD/JPY",
        start_date="2022-01-01",
        end_date="2025-01-01"
    )

    logger.info(f"\nUSD/JPY データ:")
    logger.info(f"期間: {usd_jpy.index.min()} ~ {usd_jpy.index.max()}")
    logger.info(f"件数: {len(usd_jpy)}")
    logger.info(f"\n最初の5件:\n{usd_jpy.head()}")
    logger.info(f"\n最後の5件:\n{usd_jpy.tail()}")

    # 株価指数取得
    indices = yf_data.get_stock_indices(start_date="2024-01-01")
    logger.info(f"\n株価指数取得: {list(indices.keys())}")

    return usd_jpy, indices


if __name__ == "__main__":
    test_yahoo_finance()
