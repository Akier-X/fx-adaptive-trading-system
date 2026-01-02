"""
リアルタイム自動取引ボット - 実用的世界最強システム

デモ口座で安全にテスト → 1週間検証 → 本番移行
"""
import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Optional
from dotenv import load_dotenv
import oandapyV20
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts

load_dotenv()


class LiveTradingBot:
    """リアルタイム自動取引ボット"""

    def __init__(self, pair: str = 'USD/JPY', initial_capital: float = 10000):
        """
        Args:
            pair: 通貨ペア
            initial_capital: 初期資金（円）
        """
        # OANDA API設定
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.access_token = os.getenv('OANDA_ACCESS_TOKEN')
        self.environment = os.getenv('OANDA_ENVIRONMENT', 'practice')

        if not self.account_id or not self.access_token:
            raise ValueError("OANDA API credentials not found in .env")

        # API初期化
        if self.environment == 'practice':
            self.api = API(access_token=self.access_token, environment="practice")
            logger.warning("⚠️ デモ口座モード（practice）で実行中")
        else:
            self.api = API(access_token=self.access_token)
            logger.warning("⚠️⚠️⚠️ 本番口座モード（live）で実行中 - 実際のお金を使います！")

        # 通貨ペア設定
        self.pair = pair
        self.instrument = self._convert_pair_to_oanda(pair)

        # 資金管理
        self.initial_capital = initial_capital
        self.current_capital = initial_capital

        # モデル読み込み
        self.models = self._load_models()

        # 超積極的パラメータ
        self.phase1_confidence_threshold = 0.65
        self.phase2_min_return = 0.35
        self.kelly_fraction = 0.70
        self.max_leverage = 10.0
        self.min_position_size = 0.15
        self.max_position_size = 0.40

        # 取引履歴
        self.trade_history = []
        self.current_position = None

        # ログ設定
        log_dir = Path('logs/live_trading')
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / f"trading_{datetime.now().strftime('%Y%m%d')}.log",
            rotation="1 day",
            retention="30 days"
        )

        logger.info("="*80)
        logger.info("リアルタイム自動取引ボット起動")
        logger.info("="*80)
        logger.info(f"環境: {self.environment}")
        logger.info(f"口座ID: {self.account_id}")
        logger.info(f"通貨ペア: {self.pair} ({self.instrument})")
        logger.info(f"初期資金: ¥{self.initial_capital:,.0f}")
        logger.info(f"Kelly分数: {self.kelly_fraction}")
        logger.info(f"最大レバレッジ: {self.max_leverage}x")

    def _convert_pair_to_oanda(self, pair: str) -> str:
        """通貨ペア名をOANDA形式に変換"""
        return pair.replace('/', '_')

    def _load_models(self) -> Dict:
        """モデル読み込み"""
        logger.info("\nモデル読み込み中...")

        pair_code = self.pair.replace('/', '_')

        # Phase 1.8
        phase1_path = Path(f'models/phase1_8/{pair_code}_ensemble_models.pkl')
        if not phase1_path.exists():
            raise FileNotFoundError(f"Phase 1.8モデルが見つかりません: {phase1_path}")

        phase1_data = joblib.load(phase1_path)
        logger.info(f"  ✅ Phase 1.8: 精度{phase1_data['metadata']['accuracy']:.2f}%")

        # Phase 2
        phase2_path = Path(f'models/phase2/{pair_code}_xgboost_model.pkl')
        if not phase2_path.exists():
            raise FileNotFoundError(f"Phase 2モデルが見つかりません: {phase2_path}")

        phase2_data = joblib.load(phase2_path)
        logger.info(f"  ✅ Phase 2: Sharpe {phase2_data['metadata']['sharpe']:.2f}")

        return {
            'phase1': phase1_data,
            'phase2': phase2_data
        }

    def get_current_price(self) -> Optional[Dict]:
        """現在価格取得"""
        try:
            params = {"count": 1, "granularity": "D"}
            r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
            response = self.api.request(r)

            candle = response['candles'][0]
            current_price = {
                'time': candle['time'],
                'open': float(candle['mid']['o']),
                'high': float(candle['mid']['h']),
                'low': float(candle['mid']['l']),
                'close': float(candle['mid']['c']),
                'volume': int(candle['volume'])
            }

            return current_price

        except Exception as e:
            logger.error(f"価格取得エラー: {e}")
            return None

    def get_historical_data(self, count: int = 250) -> Optional[pd.DataFrame]:
        """過去データ取得（特徴量生成用）"""
        try:
            params = {"count": count, "granularity": "D"}
            r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
            response = self.api.request(r)

            data = []
            for candle in response['candles']:
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

            logger.info(f"  過去データ取得: {len(df)}日分")
            return df

        except Exception as e:
            logger.error(f"過去データ取得エラー: {e}")
            return None

    def generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """特徴量生成（train_and_save_modelsと同じ）"""
        try:
            features = pd.DataFrame(index=df.index)

            # 基本価格
            features['close'] = df['close']
            features['open'] = df['open']
            features['high'] = df['high']
            features['low'] = df['low']
            features['volume'] = df['volume']

            # 価格比率
            features['high_close_ratio'] = (df['high'] / df['close'] - 1) * 100
            features['low_close_ratio'] = (df['low'] / df['close'] - 1) * 100
            features['high_low_range'] = (df['high'] / df['low'] - 1) * 100

            # リターン
            for period in [1, 5, 10, 20]:
                features[f'return_{period}d'] = df['close'].pct_change(period) * 100

            # SMA
            for period in [5, 10, 20, 50, 100, 200]:
                sma = df['close'].rolling(period).mean()
                features[f'sma_{period}'] = sma
                features[f'price_vs_sma_{period}'] = ((df['close'] / sma) - 1) * 100

            # EMA
            for period in [12, 26]:
                ema = df['close'].ewm(span=period).mean()
                features[f'ema_{period}'] = ema

            # RSI
            for period in [7, 14, 21]:
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(period).mean()
                loss = -delta.where(delta < 0, 0).rolling(period).mean()
                rs = gain / loss
                features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            features['macd'] = ema12 - ema26

            # ボラティリティ
            for period in [10, 20, 50]:
                features[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * 100

            # ATR
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features['atr_14'] = true_range.rolling(14).mean()

            # ボリンジャーバンド
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_position'] = ((df['close'] - features['bb_lower']) /
                                       (features['bb_upper'] - features['bb_lower']) * 100)

            # ストキャスティクス
            for period in [14, 21]:
                lowest_low = df['low'].rolling(period).min()
                highest_high = df['high'].rolling(period).max()
                features[f'stoch_{period}'] = ((df['close'] - lowest_low) /
                                              (highest_high - lowest_low) * 100)

            # モメンタム
            for period in [10, 20]:
                features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

            # ROC
            for period in [10, 20]:
                features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                            df['close'].shift(period) * 100)

            # NaN除去
            features = features.dropna()

            return features.tail(1)  # 最新1行のみ

        except Exception as e:
            logger.error(f"特徴量生成エラー: {e}")
            return None

    def predict_signal(self, features: pd.DataFrame) -> Optional[Dict]:
        """予測シグナル生成"""
        try:
            # 特徴量抽出
            feature_cols = [col for col in features.columns]
            X = features[feature_cols].values

            # Phase 1.8予測
            ensemble_probs = np.zeros((len(X), 2))
            for name, model in self.models['phase1']['models'].items():
                X_scaled = self.models['phase1']['scaler'].transform(X)
                probs = model.predict_proba(X_scaled)
                ensemble_probs += probs * self.models['phase1']['weights'][name]

            phase1_direction = ensemble_probs.argmax(axis=1)[0]
            phase1_confidence = ensemble_probs.max(axis=1)[0]

            # Phase 2予測
            X_scaled = self.models['phase2']['scaler'].transform(X)
            phase2_expected_return = self.models['phase2']['model'].predict(X_scaled)[0]
            phase2_direction = 1 if phase2_expected_return > 0 else 0

            # ハイブリッド判定
            cond1 = phase1_confidence >= self.phase1_confidence_threshold
            cond2 = abs(phase2_expected_return) >= self.phase2_min_return
            cond3 = phase1_direction == phase2_direction

            should_trade = cond1 and cond2 and cond3

            signal = {
                'should_trade': should_trade,
                'direction': phase1_direction,
                'confidence': phase1_confidence,
                'expected_return': phase2_expected_return,
                'timestamp': datetime.now()
            }

            logger.info(f"\n予測シグナル:")
            logger.info(f"  Phase 1信頼度: {phase1_confidence:.4f}")
            logger.info(f"  Phase 2期待リターン: {phase2_expected_return:.4f}%")
            logger.info(f"  方向: {'上昇' if phase1_direction == 1 else '下降'}")
            logger.info(f"  取引判定: {'✅ 実行' if should_trade else '❌ 見送り'}")

            return signal

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            return None

    def calculate_position_size(self, signal: Dict) -> float:
        """ポジションサイズ計算"""
        confidence = signal['confidence']
        expected_return = abs(signal['expected_return'])

        # Kelly基準
        kelly_position = self.kelly_fraction * (2 * confidence - 1)
        position_size = np.clip(kelly_position, self.min_position_size, self.max_position_size)

        # レバレッジ
        if confidence >= 0.65:
            remaining_conf = 1.0 - 0.65
            leverage_factor = 1 + (confidence - 0.65) * (self.max_leverage - 1) / remaining_conf
            position_size *= leverage_factor

        # 超高信頼度ブースト
        if confidence >= 0.85:
            ultra_boost = 1 + (confidence - 0.85) * 0.6
            position_size *= ultra_boost

        # 期待リターンブースト
        if expected_return >= 0.6:
            position_size *= 1.20

        # 資金に対する比率
        position_value = self.current_capital * position_size

        logger.info(f"\nポジションサイズ計算:")
        logger.info(f"  Kelly分数: {kelly_position:.4f}")
        logger.info(f"  最終サイズ倍率: {position_size:.4f}")
        logger.info(f"  ポジション金額: ¥{position_value:,.0f}")

        return position_value

    def execute_trade(self, signal: Dict, position_value: float):
        """取引実行"""
        if not signal['should_trade']:
            logger.info("取引見送り")
            return

        try:
            # 現在価格取得
            current_price_data = self.get_current_price()
            if not current_price_data:
                logger.error("現在価格取得失敗")
                return

            current_price = current_price_data['close']

            # ユニット数計算
            units = int(position_value / current_price)
            if signal['direction'] == 0:  # 下降予測 = ショート
                units = -units

            logger.info(f"\n取引実行:")
            logger.info(f"  方向: {'ロング' if units > 0 else 'ショート'}")
            logger.info(f"  ユニット数: {abs(units)}")
            logger.info(f"  価格: {current_price}")

            # 注文データ
            order_data = {
                "order": {
                    "units": str(units),
                    "instrument": self.instrument,
                    "timeInForce": "FOK",  # Fill or Kill
                    "type": "MARKET",
                    "positionFill": "DEFAULT"
                }
            }

            # 注文実行
            r = orders.OrderCreate(accountID=self.account_id, data=order_data)
            response = self.api.request(r)

            logger.success(f"✅ 注文成功: {response}")

            # 取引履歴に記録
            trade_record = {
                'timestamp': signal['timestamp'],
                'direction': 'LONG' if units > 0 else 'SHORT',
                'units': abs(units),
                'price': current_price,
                'confidence': signal['confidence'],
                'expected_return': signal['expected_return'],
                'position_value': position_value,
                'response': response
            }
            self.trade_history.append(trade_record)

            # ポジション保存
            self.current_position = trade_record

        except Exception as e:
            logger.error(f"取引実行エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def get_account_summary(self) -> Optional[Dict]:
        """口座情報取得"""
        try:
            r = accounts.AccountSummary(accountID=self.account_id)
            response = self.api.request(r)
            return response['account']
        except Exception as e:
            logger.error(f"口座情報取得エラー: {e}")
            return None

    def run_once(self):
        """1回の取引サイクル実行"""
        logger.info("\n" + "="*80)
        logger.info(f"取引サイクル開始: {datetime.now()}")
        logger.info("="*80)

        # 1. 口座情報確認
        account = self.get_account_summary()
        if account:
            logger.info(f"\n口座残高: {account['balance']} {account['currency']}")
            logger.info(f"未実現損益: {account.get('unrealizedPL', 0)}")

        # 2. 過去データ取得
        logger.info("\n過去データ取得中...")
        historical_data = self.get_historical_data()
        if historical_data is None:
            logger.error("過去データ取得失敗")
            return

        # 3. 特徴量生成
        logger.info("特徴量生成中...")
        features = self.generate_features(historical_data)
        if features is None:
            logger.error("特徴量生成失敗")
            return

        # 4. 予測シグナル生成
        logger.info("予測シグナル生成中...")
        signal = self.predict_signal(features)
        if signal is None:
            logger.error("予測失敗")
            return

        # 5. ポジションサイズ計算
        if signal['should_trade']:
            position_value = self.calculate_position_size(signal)

            # 6. 取引実行
            self.execute_trade(signal, position_value)
        else:
            logger.info("取引見送り（条件不一致）")

        logger.info("="*80)
        logger.info("取引サイクル完了")
        logger.info("="*80 + "\n")

    def run_continuous(self, check_interval_minutes: int = 60):
        """連続実行（1時間ごと）"""
        logger.info(f"\n連続実行モード開始（{check_interval_minutes}分ごと）")
        logger.info("Ctrl+Cで停止")

        while True:
            try:
                self.run_once()

                # 次回実行まで待機
                next_run = datetime.now() + timedelta(minutes=check_interval_minutes)
                logger.info(f"\n次回実行: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{check_interval_minutes}分間待機中...\n")

                time.sleep(check_interval_minutes * 60)

            except KeyboardInterrupt:
                logger.warning("\n\nユーザーによる停止")
                break
            except Exception as e:
                logger.error(f"エラー発生: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("5分後に再試行...")
                time.sleep(300)

        logger.info("\nボット停止")
        self.print_summary()

    def print_summary(self):
        """取引サマリー表示"""
        if not self.trade_history:
            logger.info("取引履歴なし")
            return

        logger.info("\n" + "="*80)
        logger.info("取引サマリー")
        logger.info("="*80)
        logger.info(f"総取引数: {len(self.trade_history)}回")

        for i, trade in enumerate(self.trade_history, 1):
            logger.info(f"\n取引 #{i}:")
            logger.info(f"  時刻: {trade['timestamp']}")
            logger.info(f"  方向: {trade['direction']}")
            logger.info(f"  ユニット: {trade['units']}")
            logger.info(f"  価格: {trade['price']}")
            logger.info(f"  信頼度: {trade['confidence']:.4f}")


if __name__ == '__main__':
    # 初期資金1万円でボット起動
    bot = LiveTradingBot(pair='USD/JPY', initial_capital=10000)

    # 1回だけ実行（テスト用）
    bot.run_once()

    # 連続実行（1時間ごと、実運用）
    # bot.run_continuous(check_interval_minutes=60)
