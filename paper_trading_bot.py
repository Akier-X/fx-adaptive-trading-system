"""
ペーパートレーディングボット - 完全シミュレーション

実際のお金を使わず、リアルタイムデータで取引をシミュレート
1日1万円の初期資金で1週間テスト
"""
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta
from loguru import logger
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from typing import Dict, Optional
import json


class PaperTradingBot:
    """ペーパートレーディングボット（シミュレーション）"""

    def __init__(self, pair: str = 'USD/JPY', initial_capital: float = 10000):
        """
        Args:
            pair: 通貨ペア
            initial_capital: 初期資金（円）
        """
        self.pair = pair
        self.yahoo_symbol = self._convert_pair_to_yahoo(pair)

        # 資金管理
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.position = None  # {'units': 100, 'entry_price': 150.50, 'direction': 'LONG'}

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
        self.daily_equity = []

        # ログ設定
        log_dir = Path('logs/paper_trading')
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            self.log_file,
            rotation="1 day",
            retention="30 days"
        )

        logger.info("="*80)
        logger.info("ペーパートレーディングボット起動（完全シミュレーション）")
        logger.info("="*80)
        logger.info("⚠️ 実際のお金は使用しません - 安全なシミュレーション")
        logger.info(f"通貨ペア: {self.pair}")
        logger.info(f"初期資金: ¥{self.initial_capital:,.0f}")
        logger.info(f"Kelly分数: {self.kelly_fraction}")
        logger.info(f"最大レバレッジ: {self.max_leverage}x")
        logger.info("="*80)

    def _convert_pair_to_yahoo(self, pair: str) -> str:
        """通貨ペア名をYahoo Finance形式に変換"""
        return pair.replace('/', '') + '=X'

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
        logger.info(f"  ✅ Phase 2: Sharpe {phase2_data['metadata']['sharpe']:.2f}\n")

        return {
            'phase1': phase1_data,
            'phase2': phase2_data
        }

    def get_current_price(self) -> Optional[float]:
        """現在価格取得（Yahoo Finance）"""
        try:
            ticker = yf.Ticker(self.yahoo_symbol)
            data = ticker.history(period='1d', interval='1m')

            if data.empty:
                logger.warning("リアルタイムデータ取得失敗、日次データを使用")
                data = ticker.history(period='5d')

            if not data.empty:
                current_price = data['Close'].iloc[-1]
                return float(current_price)
            else:
                logger.error("価格データ取得失敗")
                return None

        except Exception as e:
            logger.error(f"価格取得エラー: {e}")
            return None

    def get_historical_data(self, days: int = 250) -> Optional[pd.DataFrame]:
        """過去データ取得"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50)

            ticker = yf.Ticker(self.yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.error("過去データ取得失敗")
                return None

            # カラム名を小文字に
            df.columns = [col.lower() for col in df.columns]
            df = df[['open', 'high', 'low', 'close', 'volume']]

            logger.info(f"  過去データ取得: {len(df)}日分")
            return df

        except Exception as e:
            logger.error(f"過去データ取得エラー: {e}")
            return None

    def generate_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """特徴量生成"""
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

            logger.info(f"\n📊 予測シグナル:")
            logger.info(f"  Phase 1信頼度: {phase1_confidence:.4f}")
            logger.info(f"  Phase 2期待リターン: {phase2_expected_return:.4f}%")
            logger.info(f"  方向: {'上昇🔺' if phase1_direction == 1 else '下降🔻'}")
            logger.info(f"  取引判定: {'✅ 実行' if should_trade else '❌ 見送り'}")

            return signal

        except Exception as e:
            logger.error(f"予測エラー: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def calculate_position_size(self, signal: Dict, current_price: float) -> int:
        """ポジションサイズ計算（ユニット数）"""
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

        # 資金に対する金額
        position_value = self.cash * position_size

        # ユニット数計算
        units = int(position_value / current_price)

        logger.info(f"\n💰 ポジションサイズ計算:")
        logger.info(f"  Kelly基準: {kelly_position:.4f}")
        logger.info(f"  最終サイズ倍率: {position_size:.4f}")
        logger.info(f"  ポジション金額: ¥{position_value:,.0f}")
        logger.info(f"  ユニット数: {units}")

        return units

    def execute_trade(self, signal: Dict, current_price: float):
        """取引実行（シミュレーション）"""
        if not signal['should_trade']:
            logger.info("\n❌ 取引見送り")
            return

        # 既存ポジションをクローズ
        if self.position:
            self._close_position(current_price)

        # 新規ポジションオープン
        units = self.calculate_position_size(signal, current_price)

        if units == 0:
            logger.warning("ユニット数0 - 取引スキップ")
            return

        direction = 'LONG' if signal['direction'] == 1 else 'SHORT'

        self.position = {
            'units': units,
            'entry_price': current_price,
            'direction': direction,
            'entry_time': datetime.now(),
            'confidence': signal['confidence'],
            'expected_return': signal['expected_return']
        }

        # キャッシュから差し引き
        position_value = units * current_price
        self.cash -= position_value

        logger.success(f"\n✅ ポジションオープン:")
        logger.success(f"  方向: {direction}")
        logger.success(f"  ユニット数: {units}")
        logger.success(f"  エントリー価格: ¥{current_price:.2f}")
        logger.success(f"  ポジション金額: ¥{position_value:,.0f}")
        logger.success(f"  残キャッシュ: ¥{self.cash:,.0f}")

    def _close_position(self, current_price: float):
        """ポジションクローズ"""
        if not self.position:
            return

        units = self.position['units']
        entry_price = self.position['entry_price']
        direction = self.position['direction']

        # 損益計算
        if direction == 'LONG':
            pnl = units * (current_price - entry_price)
        else:  # SHORT
            pnl = units * (entry_price - current_price)

        pnl_pct = (pnl / (units * entry_price)) * 100

        # キャッシュに戻す
        position_value = units * current_price
        self.cash += position_value + pnl

        # 現在の評価額更新
        self.current_capital = self.cash

        logger.info(f"\n🔄 ポジションクローズ:")
        logger.info(f"  方向: {direction}")
        logger.info(f"  エントリー: ¥{entry_price:.2f}")
        logger.info(f"  エグジット: ¥{current_price:.2f}")
        logger.info(f"  損益: ¥{pnl:,.0f} ({pnl_pct:+.2f}%)")
        logger.info(f"  新キャッシュ: ¥{self.cash:,.0f}")

        # 履歴に記録
        trade_record = {
            'entry_time': self.position['entry_time'],
            'exit_time': datetime.now(),
            'direction': direction,
            'units': units,
            'entry_price': entry_price,
            'exit_price': current_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'confidence': self.position['confidence'],
            'expected_return': self.position['expected_return']
        }
        self.trade_history.append(trade_record)

        # ポジションクリア
        self.position = None

    def update_equity(self, current_price: float):
        """評価額更新"""
        total_equity = self.cash

        if self.position:
            units = self.position['units']
            entry_price = self.position['entry_price']
            direction = self.position['direction']

            if direction == 'LONG':
                unrealized_pnl = units * (current_price - entry_price)
            else:
                unrealized_pnl = units * (entry_price - current_price)

            position_value = units * current_price
            total_equity = self.cash + position_value + unrealized_pnl

        self.current_capital = total_equity

        self.daily_equity.append({
            'timestamp': datetime.now(),
            'equity': total_equity,
            'cash': self.cash,
            'position_value': total_equity - self.cash if self.position else 0
        })

    def run_once(self):
        """1回の取引サイクル実行"""
        logger.info("\n" + "="*80)
        logger.info(f"🤖 取引サイクル開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

        # 1. 現在価格取得
        logger.info("\n📈 現在価格取得中...")
        current_price = self.get_current_price()
        if current_price is None:
            logger.error("価格取得失敗")
            return

        logger.info(f"  現在価格: ¥{current_price:.2f}")

        # 2. 評価額更新
        self.update_equity(current_price)
        logger.info(f"  現在評価額: ¥{self.current_capital:,.0f}")
        logger.info(f"  損益: ¥{self.current_capital - self.initial_capital:+,.0f} ({(self.current_capital/self.initial_capital - 1)*100:+.2f}%)")

        # 3. 過去データ取得
        logger.info("\n📊 過去データ取得中...")
        historical_data = self.get_historical_data()
        if historical_data is None:
            return

        # 4. 特徴量生成
        logger.info("🔧 特徴量生成中...")
        features = self.generate_features(historical_data)
        if features is None:
            return

        # 5. 予測シグナル生成
        signal = self.predict_signal(features)
        if signal is None:
            return

        # 6. 取引実行
        self.execute_trade(signal, current_price)

        logger.info("\n" + "="*80)
        logger.info("✅ 取引サイクル完了")
        logger.info("="*80 + "\n")

    def run_continuous(self, check_interval_minutes: int = 60, duration_days: int = 7):
        """連続実行（テスト期間）"""
        logger.info(f"\n🚀 連続実行モード開始")
        logger.info(f"  チェック間隔: {check_interval_minutes}分")
        logger.info(f"  実行期間: {duration_days}日間")
        logger.info("  Ctrl+Cで停止\n")

        end_time = datetime.now() + timedelta(days=duration_days)
        cycle_count = 0

        while datetime.now() < end_time:
            try:
                cycle_count += 1
                logger.info(f"\n{'='*80}")
                logger.info(f"サイクル #{cycle_count}")
                logger.info(f"残り時間: {end_time - datetime.now()}")
                logger.info(f"{'='*80}")

                self.run_once()

                # 次回実行まで待機
                if datetime.now() < end_time:
                    next_run = datetime.now() + timedelta(minutes=check_interval_minutes)
                    if next_run > end_time:
                        break

                    logger.info(f"\n⏰ 次回実行: {next_run.strftime('%Y-%m-%d %H:%M:%S')}")
                    logger.info(f"💤 {check_interval_minutes}分間待機中...\n")

                    time.sleep(check_interval_minutes * 60)

            except KeyboardInterrupt:
                logger.warning("\n\n⚠️ ユーザーによる停止")
                break
            except Exception as e:
                logger.error(f"❌ エラー発生: {e}")
                import traceback
                logger.error(traceback.format_exc())
                logger.info("⏳ 5分後に再試行...")
                time.sleep(300)

        logger.success("\n\n🏁 テスト期間終了")
        self.print_summary()
        self.save_results()

    def print_summary(self):
        """取引サマリー表示"""
        logger.info("\n" + "="*80)
        logger.info("📊 最終サマリー")
        logger.info("="*80)

        logger.info(f"\n💰 資金:")
        logger.info(f"  初期資金: ¥{self.initial_capital:,.0f}")
        logger.info(f"  最終評価額: ¥{self.current_capital:,.0f}")
        logger.info(f"  総損益: ¥{self.current_capital - self.initial_capital:+,.0f}")
        logger.info(f"  リターン: {(self.current_capital/self.initial_capital - 1)*100:+.2f}%")

        if self.trade_history:
            logger.info(f"\n📈 取引統計:")
            logger.info(f"  総取引数: {len(self.trade_history)}回")

            wins = [t for t in self.trade_history if t['pnl'] > 0]
            losses = [t for t in self.trade_history if t['pnl'] <= 0]

            logger.info(f"  勝ちトレード: {len(wins)}回")
            logger.info(f"  負けトレード: {len(losses)}回")
            logger.info(f"  勝率: {len(wins)/len(self.trade_history)*100:.2f}%")

            if wins:
                avg_win = np.mean([t['pnl'] for t in wins])
                logger.info(f"  平均勝ちトレード: ¥{avg_win:,.0f}")

            if losses:
                avg_loss = np.mean([t['pnl'] for t in losses])
                logger.info(f"  平均負けトレード: ¥{avg_loss:,.0f}")

            total_win_pnl = sum([t['pnl'] for t in wins]) if wins else 0
            total_loss_pnl = abs(sum([t['pnl'] for t in losses])) if losses else 1

            pf = total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else 0
            logger.info(f"  プロフィットファクター: {pf:.2f}")

        logger.info("="*80 + "\n")

    def save_results(self):
        """結果保存"""
        results_dir = Path('outputs/paper_trading')
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 取引履歴保存
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_file = results_dir / f'trades_{timestamp}.csv'
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
            logger.info(f"📄 取引履歴保存: {trades_file}")

        # 日次評価額保存
        if self.daily_equity:
            equity_df = pd.DataFrame(self.daily_equity)
            equity_file = results_dir / f'equity_{timestamp}.csv'
            equity_df.to_csv(equity_file, index=False, encoding='utf-8-sig')
            logger.info(f"📄 評価額推移保存: {equity_file}")

        # サマリー保存
        summary = {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_pnl': self.current_capital - self.initial_capital,
            'return_pct': (self.current_capital/self.initial_capital - 1)*100,
            'total_trades': len(self.trade_history),
            'log_file': str(self.log_file)
        }

        summary_file = results_dir / f'summary_{timestamp}.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 サマリー保存: {summary_file}\n")


if __name__ == '__main__':
    # 初期資金1万円でボット起動
    bot = PaperTradingBot(pair='USD/JPY', initial_capital=10000)

    # テストモード（1回だけ実行）
    bot.run_once()

    # 実運用テスト（1週間、1時間ごと）
    # bot.run_continuous(check_interval_minutes=60, duration_days=7)
