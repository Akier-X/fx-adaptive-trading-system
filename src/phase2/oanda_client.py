"""
OANDA API Client for Phase 2 Live Trading

リアルタイムデータ取得・注文執行
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.pricing import PricingStream, PricingInfo
from oandapyV20.endpoints.instruments import InstrumentsCandles
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.positions import PositionList, PositionDetails
from oandapyV20.endpoints.accounts import AccountDetails
import warnings
warnings.filterwarnings('ignore')


class OANDAClient:
    """
    OANDA API クライアント

    機能:
    1. リアルタイム価格ストリーミング
    2. ヒストリカルデータ取得（1分足）
    3. 注文執行（成行・指値・逆指値）
    4. ポジション管理
    5. アカウント情報取得

    目標レイテンシ: < 100ms
    """

    def __init__(
        self,
        account_id: Optional[str] = None,
        access_token: Optional[str] = None,
        environment: str = 'practice'
    ):
        """
        Parameters:
        -----------
        account_id : str, optional
            OANDA アカウントID（環境変数から自動取得可能）
        access_token : str, optional
            OANDA アクセストークン（環境変数から自動取得可能）
        environment : str
            'practice' または 'live'
        """
        # 環境変数から取得
        self.account_id = account_id or os.getenv('OANDA_ACCOUNT_ID')
        self.access_token = access_token or os.getenv('OANDA_ACCESS_TOKEN')
        self.environment = environment or os.getenv('OANDA_ENVIRONMENT', 'practice')

        if not self.account_id or not self.access_token:
            raise ValueError(
                "OANDA credentials not found. "
                "Set OANDA_ACCOUNT_ID and OANDA_ACCESS_TOKEN environment variables."
            )

        # APIクライアント初期化
        self.api = API(
            access_token=self.access_token,
            environment=self.environment
        )

        # 対応通貨ペア
        self.instruments = [
            'USD_JPY',
            'EUR_USD',
            'GBP_USD',
            'AUD_USD',
            'USD_CAD',
            'USD_CHF',
            'EUR_JPY',
            'GBP_JPY',
        ]

        logger.info(f"OANDA Client initialized - {self.environment}")
        logger.info(f"Account: {self.account_id}")
        logger.info(f"Instruments: {len(self.instruments)}")

    def get_current_price(
        self,
        instrument: str
    ) -> Dict[str, float]:
        """
        現在価格取得（リアルタイム）

        Parameters:
        -----------
        instrument : str
            通貨ペア（例: "USD_JPY"）

        Returns:
        --------
        Dict[str, float]
            {
                'bid': ビッド価格,
                'ask': アスク価格,
                'mid': 中値,
                'spread': スプレッド,
                'time': タイムスタンプ
            }
        """
        try:
            params = {"instruments": instrument}
            request = PricingInfo(accountID=self.account_id, params=params)

            response = self.api.request(request)

            if 'prices' in response and len(response['prices']) > 0:
                price_data = response['prices'][0]

                bid = float(price_data['bids'][0]['price'])
                ask = float(price_data['asks'][0]['price'])
                mid = (bid + ask) / 2
                spread = ask - bid

                return {
                    'bid': bid,
                    'ask': ask,
                    'mid': mid,
                    'spread': spread,
                    'time': datetime.fromisoformat(price_data['time'].replace('Z', '+00:00'))
                }
            else:
                logger.error(f"No price data for {instrument}")
                return None

        except Exception as e:
            logger.error(f"Error getting current price for {instrument}: {e}")
            return None

    def get_historical_candles(
        self,
        instrument: str,
        granularity: str = 'M1',
        count: int = 500
    ) -> pd.DataFrame:
        """
        ヒストリカルデータ取得

        Parameters:
        -----------
        instrument : str
            通貨ペア
        granularity : str
            時間足（M1, M5, M15, H1, H4, D）
        count : int
            取得本数（最大5000）

        Returns:
        --------
        pd.DataFrame
            OHLCV データ
        """
        try:
            params = {
                "granularity": granularity,
                "count": count
            }

            request = InstrumentsCandles(instrument=instrument, params=params)
            response = self.api.request(request)

            # データフレーム変換
            candles = []
            for candle in response['candles']:
                if candle['complete']:
                    candles.append({
                        'time': datetime.fromisoformat(candle['time'].replace('Z', '+00:00')),
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })

            df = pd.DataFrame(candles)
            df.set_index('time', inplace=True)

            logger.info(f"Retrieved {len(df)} candles for {instrument} ({granularity})")

            return df

        except Exception as e:
            logger.error(f"Error getting historical candles: {e}")
            return pd.DataFrame()

    def create_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        成行注文作成

        Parameters:
        -----------
        instrument : str
            通貨ペア
        units : int
            ユニット数（正=買い、負=売り）
        stop_loss : float, optional
            ストップロス価格
        take_profit : float, optional
            テイクプロフィット価格

        Returns:
        --------
        Dict
            注文結果
        """
        try:
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",  # Fill or Kill
                    "positionFill": "DEFAULT"
                }
            }

            # ストップロス設定
            if stop_loss is not None:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss)
                }

            # テイクプロフィット設定
            if take_profit is not None:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit)
                }

            request = OrderCreate(accountID=self.account_id, data=order_data)
            response = self.api.request(request)

            logger.info(f"Market order created: {instrument} {units} units")

            return response

        except Exception as e:
            logger.error(f"Error creating market order: {e}")
            return None

    def get_open_positions(self) -> List[Dict]:
        """
        オープンポジション取得

        Returns:
        --------
        List[Dict]
            ポジションリスト
        """
        try:
            request = PositionList(accountID=self.account_id)
            response = self.api.request(request)

            positions = []
            for position in response.get('positions', []):
                if float(position['long']['units']) != 0 or float(position['short']['units']) != 0:
                    positions.append({
                        'instrument': position['instrument'],
                        'long_units': float(position['long']['units']),
                        'short_units': float(position['short']['units']),
                        'unrealized_pl': float(position['unrealizedPL'])
                    })

            return positions

        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def get_account_summary(self) -> Dict:
        """
        アカウント情報取得

        Returns:
        --------
        Dict
            {
                'balance': 口座残高,
                'unrealized_pl': 未実現損益,
                'nav': 純資産,
                'margin_used': 使用証拠金,
                'margin_available': 利用可能証拠金
            }
        """
        try:
            request = AccountDetails(accountID=self.account_id)
            response = self.api.request(request)

            account = response['account']

            return {
                'balance': float(account['balance']),
                'unrealized_pl': float(account['unrealizedPL']),
                'nav': float(account['NAV']),
                'margin_used': float(account['marginUsed']),
                'margin_available': float(account['marginAvailable']),
                'currency': account['currency']
            }

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return None

    def close_position(
        self,
        instrument: str,
        side: str = 'ALL'
    ) -> Dict:
        """
        ポジションクローズ

        Parameters:
        -----------
        instrument : str
            通貨ペア
        side : str
            'LONG', 'SHORT', または 'ALL'

        Returns:
        --------
        Dict
            クローズ結果
        """
        try:
            # 現在のポジション取得
            positions = self.get_open_positions()

            for position in positions:
                if position['instrument'] == instrument:
                    # ロングポジションクローズ
                    if side in ['LONG', 'ALL'] and position['long_units'] > 0:
                        units = -int(position['long_units'])
                        self.create_market_order(instrument, units)

                    # ショートポジションクローズ
                    if side in ['SHORT', 'ALL'] and position['short_units'] < 0:
                        units = -int(position['short_units'])
                        self.create_market_order(instrument, units)

            logger.info(f"Position closed: {instrument} ({side})")

            return {'status': 'success'}

        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {'status': 'error', 'message': str(e)}

    def stream_prices(
        self,
        instruments: List[str],
        callback: callable
    ):
        """
        価格ストリーミング（リアルタイム）

        Parameters:
        -----------
        instruments : List[str]
            通貨ペアリスト
        callback : callable
            価格更新時のコールバック関数
            callback(instrument, price_data)
        """
        try:
            params = {
                "instruments": ",".join(instruments)
            }

            request = PricingStream(accountID=self.account_id, params=params)

            logger.info(f"Starting price stream for {len(instruments)} instruments")

            for response in self.api.request(request):
                if 'type' in response and response['type'] == 'PRICE':
                    instrument = response['instrument']

                    bid = float(response['bids'][0]['price'])
                    ask = float(response['asks'][0]['price'])

                    price_data = {
                        'bid': bid,
                        'ask': ask,
                        'mid': (bid + ask) / 2,
                        'spread': ask - bid,
                        'time': datetime.fromisoformat(response['time'].replace('Z', '+00:00'))
                    }

                    # コールバック実行
                    callback(instrument, price_data)

        except Exception as e:
            logger.error(f"Error in price streaming: {e}")


if __name__ == "__main__":
    # 動作テスト
    client = OANDAClient()

    # 現在価格取得
    price = client.get_current_price('USD_JPY')
    logger.info(f"Current USD/JPY: {price}")

    # ヒストリカルデータ取得
    df = client.get_historical_candles('USD_JPY', 'M1', 100)
    logger.info(f"Historical data shape: {df.shape}")

    # アカウント情報
    account = client.get_account_summary()
    logger.info(f"Account balance: {account['balance']} {account['currency']}")
