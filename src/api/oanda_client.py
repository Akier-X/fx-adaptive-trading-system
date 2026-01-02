"""
Oanda API クライアント
"""
from oandapyV20 import API
from oandapyV20.endpoints import accounts, orders, pricing, instruments
from oandapyV20.exceptions import V20Error
from loguru import logger
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta

from ..config import settings


class OandaClient:
    """Oanda APIクライアント"""

    def __init__(self):
        self.client = API(
            access_token=settings.oanda_access_token,
            environment=settings.oanda_environment
        )
        self.account_id = settings.oanda_account_id

    def get_account_summary(self) -> Dict:
        """アカウント情報を取得"""
        try:
            endpoint = accounts.AccountSummary(accountID=self.account_id)
            response = self.client.request(endpoint)
            return response
        except V20Error as e:
            logger.error(f"アカウント情報の取得に失敗: {e}")
            raise

    def get_balance(self) -> float:
        """現在の残高を取得"""
        summary = self.get_account_summary()
        return float(summary['account']['balance'])

    def get_current_price(self, instrument: str = "USD_JPY") -> Dict:
        """現在価格を取得"""
        try:
            params = {"instruments": instrument}
            endpoint = pricing.PricingInfo(
                accountID=self.account_id,
                params=params
            )
            response = self.client.request(endpoint)
            return response['prices'][0]
        except V20Error as e:
            logger.error(f"価格取得に失敗: {e}")
            raise

    def get_historical_data(
        self,
        instrument: str = "USD_JPY",
        granularity: str = "H1",
        count: int = 500
    ) -> pd.DataFrame:
        """
        過去データを取得
        
        Args:
            instrument: 通貨ペア (例: USD_JPY, EUR_USD)
            granularity: 時間足 (M1, M5, H1, D など)
            count: 取得するローソク足の数
        """
        try:
            params = {
                "granularity": granularity,
                "count": count
            }
            endpoint = instruments.InstrumentsCandles(
                instrument=instrument,
                params=params
            )
            response = self.client.request(endpoint)
            
            # DataFrameに変換
            candles = response['candles']
            data = []
            for candle in candles:
                if candle['complete']:
                    data.append({
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': int(candle['volume'])
                    })
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
            
        except V20Error as e:
            logger.error(f"履歴データの取得に失敗: {e}")
            raise

    def place_market_order(
        self,
        instrument: str,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict:
        """
        成行注文を発注
        
        Args:
            instrument: 通貨ペア
            units: ユニット数 (正: 買い、負: 売り)
            stop_loss: ストップロス価格
            take_profit: テイクプロフィット価格
        """
        try:
            order_data = {
                "order": {
                    "type": "MARKET",
                    "instrument": instrument,
                    "units": str(units),
                    "timeInForce": "FOK",
                    "positionFill": "DEFAULT"
                }
            }
            
            # ストップロス設定
            if stop_loss:
                order_data["order"]["stopLossOnFill"] = {
                    "price": str(stop_loss)
                }
            
            # テイクプロフィット設定
            if take_profit:
                order_data["order"]["takeProfitOnFill"] = {
                    "price": str(take_profit)
                }
            
            endpoint = orders.OrderCreate(
                accountID=self.account_id,
                data=order_data
            )
            response = self.client.request(endpoint)
            logger.info(f"注文成功: {response}")
            return response
            
        except V20Error as e:
            logger.error(f"注文失敗: {e}")
            raise

    def get_open_positions(self) -> List[Dict]:
        """オープンポジションを取得"""
        try:
            endpoint = accounts.AccountDetails(accountID=self.account_id)
            response = self.client.request(endpoint)
            return response['account'].get('positions', [])
        except V20Error as e:
            logger.error(f"ポジション取得に失敗: {e}")
            raise

    def close_position(self, instrument: str) -> Dict:
        """ポジションをクローズ"""
        try:
            # まず現在のポジションを確認
            positions = self.get_open_positions()
            position = next(
                (p for p in positions if p['instrument'] == instrument),
                None
            )
            
            if not position:
                logger.warning(f"{instrument}のポジションが見つかりません")
                return {}
            
            # ロング/ショートに応じてクローズ
            long_units = int(float(position.get('long', {}).get('units', 0)))
            short_units = int(float(position.get('short', {}).get('units', 0)))
            
            if long_units > 0:
                return self.place_market_order(instrument, -long_units)
            elif short_units < 0:
                return self.place_market_order(instrument, -short_units)
            
            return {}
            
        except V20Error as e:
            logger.error(f"ポジションクローズに失敗: {e}")
            raise
