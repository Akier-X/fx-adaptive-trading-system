"""
Phase 2 リアルタイムデータパイプライン

目標性能:
- データ取得: < 100ms
- 特徴量生成: < 500ms
- 合計レイテンシ: < 1秒
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# OANDA Client
from src.phase2.oanda_client import OANDAClient

# Phase 1の最高精度データソース
from src.data_sources.yahoo_finance import YahooFinanceCollector
from src.data_sources.economic_indicators import EconomicIndicatorsCollector

# OMEGA ULTIMATEの全データソース
from src.world_class_trader.news_sentiment_integrator import NewsSentimentIntegrator
from src.world_class_trader.enhanced_data_sources import EnhancedDataSources
from src.world_class_trader.social_media_analyzer import SocialMediaAnalyzer
from src.world_class_trader.cot_analyzer import COTAnalyzer
from src.world_class_trader.geopolitical_analyzer import GeopoliticalAnalyzer

# キャッシュ
import pickle
import hashlib


class RealtimeDataPipeline:
    """
    リアルタイムデータパイプライン

    機能:
    1. OANDA APIから1分足データ取得
    2. 複数時間足（M1, M5, M15, H1, H4, D）の同期
    3. 1200+特徴量のリアルタイム生成
    4. キャッシュによる高速化
    5. 複数通貨ペア対応

    アーキテクチャ:
    OANDA API → キャッシュ → 特徴量生成 → 予測エンジンへ
    """

    def __init__(
        self,
        instruments: List[str] = None,
        cache_dir: Path = Path("data/phase2/cache")
    ):
        """
        Parameters:
        -----------
        instruments : List[str], optional
            対象通貨ペア（デフォルト: 8ペア）
        cache_dir : Path
            キャッシュディレクトリ
        """
        # 対象通貨ペア
        self.instruments = instruments or [
            'USD_JPY',
            'EUR_USD',
            'GBP_USD',
            'AUD_USD',
            'USD_CAD',
            'USD_CHF',
            'EUR_JPY',
            'GBP_JPY',
        ]

        # キャッシュディレクトリ
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # OANDAクライアント初期化
        self.oanda = OANDAClient()

        # データソース初期化（OMEGA ULTIMATE）
        logger.info("初期化中: 全データソース...")

        # Phase 1 ベースデータソース
        self.yahoo = YahooFinanceCollector()
        self.economic = EconomicIndicatorsCollector()

        # OMEGA ULTIMATE データソース
        self.news_sentiment = NewsSentimentIntegrator()
        self.enhanced_data = EnhancedDataSources()
        self.social_media = SocialMediaAnalyzer()
        self.cot = COTAnalyzer()
        self.geopolitical = GeopoliticalAnalyzer()

        # データキャッシュ
        self.price_cache = {}  # {instrument: {granularity: DataFrame}}
        self.feature_cache = {}  # {cache_key: features}

        logger.info("✅ リアルタイムパイプライン初期化完了")
        logger.info(f"  対象通貨ペア: {len(self.instruments)}")

    def fetch_realtime_data(
        self,
        instrument: str,
        granularities: List[str] = ['M1', 'M5', 'M15', 'H1', 'H4', 'D']
    ) -> Dict[str, pd.DataFrame]:
        """
        リアルタイムデータ取得（複数時間足）

        Parameters:
        -----------
        instrument : str
            通貨ペア
        granularities : List[str]
            時間足リスト

        Returns:
        --------
        Dict[str, pd.DataFrame]
            {時間足: OHLCVデータ}
        """
        data = {}

        for granularity in granularities:
            try:
                # キャッシュチェック
                cache_key = f"{instrument}_{granularity}"
                if cache_key in self.price_cache:
                    cached_df = self.price_cache[cache_key]

                    # キャッシュが新しければ使用（1分以内）
                    if not cached_df.empty:
                        last_time = cached_df.index[-1]
                        if datetime.now(last_time.tzinfo) - last_time < timedelta(minutes=1):
                            data[granularity] = cached_df
                            continue

                # OANDA APIからデータ取得
                df = self.oanda.get_historical_candles(
                    instrument=instrument,
                    granularity=granularity,
                    count=500  # 十分な履歴データ
                )

                if not df.empty:
                    # キャッシュ更新
                    self.price_cache[cache_key] = df
                    data[granularity] = df

            except Exception as e:
                logger.error(f"データ取得エラー ({instrument} {granularity}): {e}")

        logger.info(f"  ✓ {instrument}: {len(data)}時間足取得完了")

        return data

    def generate_features(
        self,
        instrument: str,
        date: datetime = None
    ) -> Dict[str, float]:
        """
        1200+特徴量生成（リアルタイム）

        Parameters:
        -----------
        instrument : str
            通貨ペア
        date : datetime, optional
            基準日時（デフォルト: 現在時刻）

        Returns:
        --------
        Dict[str, float]
            特徴量辞書（1200+個）
        """
        if date is None:
            date = datetime.now()

        # キャッシュキー生成
        cache_key = self._generate_cache_key(instrument, date)

        # キャッシュヒット
        if cache_key in self.feature_cache:
            logger.info(f"  ✓ キャッシュヒット: {instrument}")
            return self.feature_cache[cache_key]

        logger.info(f"特徴量生成開始: {instrument} @ {date}")

        features = {}

        # 1. リアルタイム価格データ取得
        price_data = self.fetch_realtime_data(instrument)

        # 2. テクニカル指標（Phase 1.8ベース）
        technical_features = self._generate_technical_features(
            price_data,
            instrument
        )
        features.update(technical_features)

        # 3. 経済指標（FRED）
        try:
            econ_features = self.economic.generate_features(instrument, date)
            features.update(econ_features)
        except Exception as e:
            logger.warning(f"経済指標取得エラー: {e}")

        # 4. ニュースセンチメント
        try:
            news_features = self.news_sentiment.generate_features(
                instrument,
                date,
                lookback_hours=24
            )
            features.update(news_features)
        except Exception as e:
            logger.warning(f"ニュース分析エラー: {e}")

        # 5. 拡張データソース（VIX、株式、コモディティ、暗号通貨）
        try:
            enhanced_features = self.enhanced_data.generate_features(
                instrument,
                date
            )
            features.update(enhanced_features)
        except Exception as e:
            logger.warning(f"拡張データ取得エラー: {e}")

        # 6. ソーシャルメディア（Twitter、Reddit、Fear & Greed）
        try:
            social_features = self.social_media.generate_features(
                instrument,
                date,
                lookback_hours=24
            )
            features.update(social_features)
        except Exception as e:
            logger.warning(f"ソーシャルメディア分析エラー: {e}")

        # 7. COTレポート
        try:
            cot_features = self.cot.generate_features(instrument, date)
            features.update(cot_features)
        except Exception as e:
            logger.warning(f"COT分析エラー: {e}")

        # 8. 地政学指標
        try:
            geo_features = self.geopolitical.generate_features(instrument, date)
            features.update(geo_features)
        except Exception as e:
            logger.warning(f"地政学分析エラー: {e}")

        # キャッシュ保存
        self.feature_cache[cache_key] = features

        logger.info(f"✅ 特徴量生成完了: {len(features)}個")

        return features

    def _generate_technical_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        instrument: str
    ) -> Dict[str, float]:
        """
        テクニカル指標特徴量生成

        複数時間足からの指標:
        - SMA, EMA（複数期間）
        - RSI, MACD, Bollinger Bands
        - ATR, ADX
        - ストキャスティクス
        - 一目均衡表
        """
        features = {}

        try:
            # M1データ（メイン）
            if 'M1' in price_data and not price_data['M1'].empty:
                df = price_data['M1'].copy()

                # 基本価格情報
                features['current_price'] = df['close'].iloc[-1]
                features['open'] = df['open'].iloc[-1]
                features['high'] = df['high'].iloc[-1]
                features['low'] = df['low'].iloc[-1]
                features['volume'] = df['volume'].iloc[-1]

                # リターン
                features['return_1m'] = (df['close'].iloc[-1] / df['close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
                features['return_5m'] = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) * 100 if len(df) >= 6 else 0
                features['return_15m'] = (df['close'].iloc[-1] / df['close'].iloc[-16] - 1) * 100 if len(df) >= 16 else 0

                # SMA
                for period in [5, 10, 20, 50, 100, 200]:
                    if len(df) >= period:
                        sma = df['close'].rolling(period).mean().iloc[-1]
                        features[f'sma_{period}'] = sma
                        features[f'price_vs_sma_{period}'] = ((df['close'].iloc[-1] / sma) - 1) * 100

                # EMA
                for period in [12, 26, 50]:
                    if len(df) >= period:
                        ema = df['close'].ewm(span=period).mean().iloc[-1]
                        features[f'ema_{period}'] = ema
                        features[f'price_vs_ema_{period}'] = ((df['close'].iloc[-1] / ema) - 1) * 100

                # RSI
                if len(df) >= 14:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi_14'] = rsi.iloc[-1]

                # MACD
                if len(df) >= 26:
                    ema12 = df['close'].ewm(span=12).mean()
                    ema26 = df['close'].ewm(span=26).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9).mean()
                    features['macd'] = macd.iloc[-1]
                    features['macd_signal'] = signal.iloc[-1]
                    features['macd_histogram'] = (macd - signal).iloc[-1]

                # Bollinger Bands
                if len(df) >= 20:
                    sma20 = df['close'].rolling(20).mean()
                    std20 = df['close'].rolling(20).std()
                    upper_band = sma20 + (2 * std20)
                    lower_band = sma20 - (2 * std20)
                    features['bb_upper'] = upper_band.iloc[-1]
                    features['bb_middle'] = sma20.iloc[-1]
                    features['bb_lower'] = lower_band.iloc[-1]
                    features['bb_width'] = ((upper_band - lower_band) / sma20).iloc[-1]
                    features['bb_position'] = ((df['close'] - lower_band) / (upper_band - lower_band)).iloc[-1]

                # ATR
                if len(df) >= 14:
                    high_low = df['high'] - df['low']
                    high_close = (df['high'] - df['close'].shift()).abs()
                    low_close = (df['low'] - df['close'].shift()).abs()
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    atr = tr.rolling(14).mean()
                    features['atr_14'] = atr.iloc[-1]
                    features['atr_ratio'] = (atr.iloc[-1] / df['close'].iloc[-1]) * 100

            # 他の時間足からの情報
            for granularity in ['M5', 'M15', 'H1', 'H4', 'D']:
                if granularity in price_data and not price_data[granularity].empty:
                    df_tf = price_data[granularity]

                    prefix = f"{granularity.lower()}_"

                    # トレンド方向
                    if len(df_tf) >= 20:
                        sma20 = df_tf['close'].rolling(20).mean().iloc[-1]
                        features[f'{prefix}trend'] = 1 if df_tf['close'].iloc[-1] > sma20 else -1

                    # モメンタム
                    if len(df_tf) >= 10:
                        features[f'{prefix}momentum'] = (df_tf['close'].iloc[-1] / df_tf['close'].iloc[-10] - 1) * 100

        except Exception as e:
            logger.error(f"テクニカル指標生成エラー: {e}")

        return features

    def _generate_cache_key(
        self,
        instrument: str,
        date: datetime
    ) -> str:
        """
        キャッシュキー生成

        1分単位でキャッシュを管理
        """
        date_str = date.strftime('%Y%m%d_%H%M')
        key = f"{instrument}_{date_str}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_current_signal(
        self,
        instrument: str
    ) -> Dict:
        """
        現在のシグナル取得（簡易版）

        Returns:
        --------
        Dict
            {
                'instrument': 通貨ペア,
                'features': 特徴量,
                'price': 現在価格,
                'timestamp': タイムスタンプ
            }
        """
        # 現在価格
        price = self.oanda.get_current_price(instrument)

        # 特徴量生成
        features = self.generate_features(instrument)

        return {
            'instrument': instrument,
            'features': features,
            'price': price,
            'timestamp': datetime.now()
        }

    def clear_cache(self):
        """キャッシュクリア"""
        self.price_cache.clear()
        self.feature_cache.clear()
        logger.info("キャッシュクリア完了")


if __name__ == "__main__":
    # 動作テスト
    logger.info("リアルタイムパイプライン テスト開始")

    pipeline = RealtimeDataPipeline(instruments=['USD_JPY'])

    # データ取得テスト
    logger.info("\n=== データ取得テスト ===")
    data = pipeline.fetch_realtime_data('USD_JPY')
    for granularity, df in data.items():
        logger.info(f"{granularity}: {len(df)} candles")

    # 特徴量生成テスト
    logger.info("\n=== 特徴量生成テスト ===")
    features = pipeline.generate_features('USD_JPY')
    logger.info(f"生成された特徴量: {len(features)}個")

    # シグナル取得テスト
    logger.info("\n=== シグナル取得テスト ===")
    signal = pipeline.get_current_signal('USD_JPY')
    logger.info(f"現在価格: {signal['price']['mid']}")
    logger.info(f"特徴量数: {len(signal['features'])}")
