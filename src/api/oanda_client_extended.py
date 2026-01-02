"""
Oanda API クライアント拡張版

追加機能:
- ポジショニングデータ（Order Book / Position Book）
- bid/askスプレッド
- 複数時間足データ
- 複数通貨ペア同時取得
"""
from oandapyV20 import API
from oandapyV20.endpoints import accounts, orders, pricing, instruments
from oandapyV20.exceptions import V20Error
from loguru import logger
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

from ..config import settings


class OandaClientExtended:
    """Oanda API拡張クライアント"""

    def __init__(self):
        self.client = API(
            access_token=settings.oanda_access_token,
            environment=settings.oanda_environment
        )
        self.account_id = settings.oanda_account_id

    def get_historical_data_with_spread(
        self,
        instrument: str = "USD_JPY",
        granularity: str = "H1",
        count: int = 500
    ) -> pd.DataFrame:
        """
        過去データをbid/ask情報付きで取得

        Args:
            instrument: 通貨ペア
            granularity: 時間足
            count: 取得数

        Returns:
            bid/ask/mid/スプレッド情報付きDataFrame
        """
        try:
            params = {
                "granularity": granularity,
                "count": count,
                "price": "MBA"  # M=mid, B=bid, A=ask
            }
            endpoint = instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
            response = self.client.request(endpoint)

            candles = response['candles']
            data = []
            for candle in candles:
                if candle['complete']:
                    mid = candle.get('mid', {})
                    bid = candle.get('bid', {})
                    ask = candle.get('ask', {})

                    mid_close = float(mid.get('c', 0))
                    bid_close = float(bid.get('c', 0))
                    ask_close = float(ask.get('c', 0))

                    data.append({
                        'time': candle['time'],
                        'open': float(mid.get('o', 0)),
                        'high': float(mid.get('h', 0)),
                        'low': float(mid.get('l', 0)),
                        'close': mid_close,
                        'volume': int(candle['volume']),
                        'bid': bid_close,
                        'ask': ask_close,
                        'spread': ask_close - bid_close,
                        'spread_pct': ((ask_close - bid_close) / mid_close * 100) if mid_close > 0 else 0
                    })

            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df

        except V20Error as e:
            logger.error(f"履歴データ取得失敗: {e}")
            raise

    def get_order_book(self, instrument: str = "USD_JPY") -> Dict:
        """
        オーダーブック（指値注文の分布）を取得

        Returns:
            オーダーブック情報
        """
        try:
            endpoint = instruments.InstrumentsOrderBook(instrument=instrument)
            response = self.client.request(endpoint)
            return response.get('orderBook', {})
        except V20Error as e:
            logger.error(f"オーダーブック取得失敗: {e}")
            return {}

    def get_position_book(self, instrument: str = "USD_JPY") -> Dict:
        """
        ポジションブック（市場参加者のポジション分布）を取得

        Returns:
            ポジションブック情報
        """
        try:
            endpoint = instruments.InstrumentsPositionBook(instrument=instrument)
            response = self.client.request(endpoint)
            return response.get('positionBook', {})
        except V20Error as e:
            logger.error(f"ポジションブック取得失敗: {e}")
            return {}

    def get_multi_timeframe_data(
        self,
        instrument: str = "USD_JPY",
        timeframes: List[str] = None,
        count: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        複数時間足のデータを同時取得

        Args:
            instrument: 通貨ペア
            timeframes: 時間足リスト（例: ["M5", "H1", "H4", "D"]）
            count: 各時間足の取得数

        Returns:
            {timeframe: DataFrame} の辞書
        """
        if timeframes is None:
            timeframes = ["M5", "H1", "H4", "D"]

        multi_data = {}

        for tf in timeframes:
            try:
                logger.info(f"{instrument} {tf} データ取得中...")
                data = self.get_historical_data_with_spread(
                    instrument=instrument,
                    granularity=tf,
                    count=count
                )
                multi_data[tf] = data
            except Exception as e:
                logger.error(f"{tf} 取得エラー: {e}")
                continue

        return multi_data

    def get_multiple_instruments(
        self,
        instruments: List[str] = None,
        granularity: str = "H1",
        count: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        複数通貨ペアのデータを同時取得

        Args:
            instruments: 通貨ペアリスト
            granularity: 時間足
            count: 取得数

        Returns:
            {instrument: DataFrame} の辞書
        """
        if instruments is None:
            instruments = ["USD_JPY", "EUR_USD", "GBP_USD", "EUR_JPY"]

        multi_instruments = {}

        for inst in instruments:
            try:
                logger.info(f"{inst} データ取得中...")
                data = self.get_historical_data_with_spread(
                    instrument=inst,
                    granularity=granularity,
                    count=count
                )
                multi_instruments[inst] = data

                import time
                time.sleep(0.3)  # API制限対策

            except Exception as e:
                logger.error(f"{inst} 取得エラー: {e}")
                continue

        return multi_instruments

    def get_positioning_features(self, instrument: str = "USD_JPY") -> Dict:
        """
        ポジショニングデータから特徴量を抽出

        Returns:
            ポジショニング関連の特徴量
        """
        try:
            order_book = self.get_order_book(instrument)
            position_book = self.get_position_book(instrument)

            features = {}

            # オーダーブックの特徴量
            if order_book:
                buckets = order_book.get('buckets', [])
                if buckets:
                    # ロング/ショートの注文比率
                    total_long = sum(float(b.get('longCountPercent', 0)) for b in buckets)
                    total_short = sum(float(b.get('shortCountPercent', 0)) for b in buckets)

                    features['order_long_pct'] = total_long / len(buckets) if buckets else 0
                    features['order_short_pct'] = total_short / len(buckets) if buckets else 0
                    features['order_bias'] = features['order_long_pct'] - features['order_short_pct']

            # ポジションブックの特徴量
            if position_book:
                buckets = position_book.get('buckets', [])
                if buckets:
                    total_long = sum(float(b.get('longCountPercent', 0)) for b in buckets)
                    total_short = sum(float(b.get('shortCountPercent', 0)) for b in buckets)

                    features['position_long_pct'] = total_long / len(buckets) if buckets else 0
                    features['position_short_pct'] = total_short / len(buckets) if buckets else 0
                    features['position_bias'] = features['position_long_pct'] - features['position_short_pct']

            return features

        except Exception as e:
            logger.error(f"ポジショニング特徴量抽出エラー: {e}")
            return {}
