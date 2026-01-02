"""
経済指標データソース

FRED API（Federal Reserve Economic Data）から経済指標を取得
"""
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from typing import Dict, List, Optional
import os


class EconomicIndicators:
    """
    経済指標の取得と特徴量化

    FRED API使用（無料）
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: FRED API キー（.envから取得可能）
        """
        # まずos.getenvを試し、次にconfigから取得
        if api_key is None:
            api_key = os.getenv('FRED_API_KEY')

        # configからも試す
        if api_key is None or api_key == 'your_fred_api_key_here' or api_key == '':
            try:
                from ..config import settings
                if settings.fred_api_key and settings.fred_api_key != 'your_fred_api_key_here':
                    api_key = settings.fred_api_key
            except:
                pass

        self.api_key = api_key
        self.fred = None

        if self.api_key and self.api_key != 'your_fred_api_key_here' and self.api_key != '':
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
                logger.info(f"FRED API 初期化成功（キー: {self.api_key[:8]}...）")
            except ImportError:
                logger.warning("fredapi パッケージがインストールされていません")
                logger.warning("pip install fredapi を実行してください")
            except Exception as e:
                logger.error(f"FRED API 初期化エラー: {e}")
        else:
            logger.warning("FRED_API_KEY が設定されていません")
            logger.warning("経済指標データは取得できません（デモデータを使用）")

    def get_interest_rates(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        主要国の政策金利を取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            金利データ
        """
        if not self.fred:
            return self._generate_demo_rates(start_date, end_date)

        try:
            # 米国政策金利（Federal Funds Rate）
            us_rate = self.fred.get_series('FEDFUNDS', start_date, end_date)

            # 日本政策金利（代替: 日本の短期金利）
            # FRED APIでは日本の政策金利は直接取得不可（基本的にゼロ金利）
            # 10年国債利回りを使用
            jp_rate = self.fred.get_series('IRLTLT01JPM156N', start_date, end_date)

            # データフレーム化
            df = pd.DataFrame({
                'us_fed_rate': us_rate,
                'jp_rate': jp_rate
            })

            # 金利差（USD_JPYの主要ドライバー）
            df['rate_differential'] = df['us_fed_rate'] - df['jp_rate']

            # 変化率
            df['us_rate_change'] = df['us_fed_rate'].diff()
            df['jp_rate_change'] = df['jp_rate'].diff()

            df = df.fillna(method='ffill').fillna(0)

            logger.info(f"金利データ取得: {len(df)}件")
            return df

        except Exception as e:
            logger.error(f"金利データ取得エラー: {e}")
            return self._generate_demo_rates(start_date, end_date)

    def get_inflation_data(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        インフレ率（CPI）を取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            インフレデータ
        """
        if not self.fred:
            return self._generate_demo_inflation(start_date, end_date)

        try:
            # 米国CPI
            us_cpi = self.fred.get_series('CPIAUCSL', start_date, end_date)

            # 日本CPI
            jp_cpi = self.fred.get_series('JPNCPIALLMINMEI', start_date, end_date)

            df = pd.DataFrame({
                'us_cpi': us_cpi,
                'jp_cpi': jp_cpi
            })

            # 前年同月比（YoY）
            df['us_cpi_yoy'] = df['us_cpi'].pct_change(12) * 100
            df['jp_cpi_yoy'] = df['jp_cpi'].pct_change(12) * 100

            df = df.fillna(method='ffill').fillna(0)

            logger.info(f"インフレデータ取得: {len(df)}件")
            return df

        except Exception as e:
            logger.error(f"インフレデータ取得エラー: {e}")
            return self._generate_demo_inflation(start_date, end_date)

    def get_unemployment_rate(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        失業率を取得

        Args:
            start_date: 開始日
            end_date: 終了日

        Returns:
            失業率データ
        """
        if not self.fred:
            return self._generate_demo_unemployment(start_date, end_date)

        try:
            # 米国失業率
            us_unemp = self.fred.get_series('UNRATE', start_date, end_date)

            # 日本失業率
            jp_unemp = self.fred.get_series('LRHUTTTTJPM156S', start_date, end_date)

            df = pd.DataFrame({
                'us_unemployment': us_unemp,
                'jp_unemployment': jp_unemp
            })

            # 変化
            df['us_unemp_change'] = df['us_unemployment'].diff()
            df['jp_unemp_change'] = df['jp_unemployment'].diff()

            df = df.fillna(method='ffill').fillna(0)

            logger.info(f"失業率データ取得: {len(df)}件")
            return df

        except Exception as e:
            logger.error(f"失業率データ取得エラー: {e}")
            return self._generate_demo_unemployment(start_date, end_date)

    def get_all_indicators(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        全経済指標を取得して統合

        Returns:
            統合された経済指標DataFrame
        """
        logger.info("経済指標データ取得中...")

        # 各指標を取得
        rates = self.get_interest_rates(start_date, end_date)
        inflation = self.get_inflation_data(start_date, end_date)
        unemployment = self.get_unemployment_rate(start_date, end_date)

        # 統合（外部結合して全期間をカバー）
        combined = rates.join(inflation, how='outer')
        combined = combined.join(unemployment, how='outer')

        # 欠損値を前方・後方埋め
        combined = combined.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"経済指標統合完了: {len(combined.columns)}個の指標")

        return combined

    def _generate_demo_rates(self, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
        """デモ用金利データ"""
        import numpy as np

        dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')
        df = pd.DataFrame(index=dates)

        df['us_fed_rate'] = 1.5 + np.random.randn(len(df)) * 0.1
        df['jp_rate'] = 0.1 + np.random.randn(len(df)) * 0.01
        df['rate_differential'] = df['us_fed_rate'] - df['jp_rate']
        df['us_rate_change'] = 0
        df['jp_rate_change'] = 0

        return df

    def _generate_demo_inflation(self, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
        """デモ用インフレデータ"""
        import numpy as np

        dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')
        df = pd.DataFrame(index=dates)

        df['us_cpi'] = 250 + np.cumsum(np.random.randn(len(df)) * 0.1)
        df['jp_cpi'] = 100 + np.cumsum(np.random.randn(len(df)) * 0.05)
        df['us_cpi_yoy'] = 2.0
        df['jp_cpi_yoy'] = 0.5

        return df

    def _generate_demo_unemployment(self, start_date: str, end_date: Optional[str]) -> pd.DataFrame:
        """デモ用失業率データ"""
        import numpy as np

        dates = pd.date_range(start=start_date, end=end_date or datetime.now(), freq='D')
        df = pd.DataFrame(index=dates)

        df['us_unemployment'] = 4.0 + np.random.randn(len(df)) * 0.1
        df['jp_unemployment'] = 2.5 + np.random.randn(len(df)) * 0.05
        df['us_unemp_change'] = 0
        df['jp_unemp_change'] = 0

        return df
