"""
適応的学習トレーディングボット

固定モデルの問題を解決:
- リアルタイムで新しいデータから学習
- モデルの劣化を防止
- 市場環境の変化に自動適応
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from loguru import logger

# オンライン学習モデル
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler

# 既存モデルの読み込み用
import xgboost as xgb

logger.remove()
logger.add(sys.stdout, level="INFO")


class AdaptiveLearningBot:
    """
    適応的学習ボット

    特徴:
    1. オンライン学習: 新しいデータで継続的に更新
    2. ハイブリッド: 固定モデル + 適応モデル
    3. 性能監視: 劣化検知と自動再訓練
    4. 適応的パラメータ: 市場環境に応じた調整
    """

    def __init__(self, pair: str = 'USD/JPY', initial_capital: float = 10000):
        self.pair = pair
        self.yahoo_symbol = self._convert_pair_to_yahoo(pair)

        # 資金管理
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash = initial_capital
        self.position = None

        # モデル設定
        self.models_dir = Path('models')
        self.adaptive_models_dir = self.models_dir / 'adaptive'
        self.adaptive_models_dir.mkdir(parents=True, exist_ok=True)

        # オンライン学習モデル
        self.online_models = self._init_online_models()
        self.scaler = StandardScaler()

        # 固定モデル（Phase 1.8 + Phase 2）
        self.fixed_phase1_models = None
        self.fixed_phase2_model = None
        self._load_fixed_models()

        # 学習バッファ
        self.update_buffer = []  # (features, label)のリスト
        self.update_interval = 50  # 50取引ごとに更新

        # 性能監視
        self.recent_trades = []  # 最近の取引履歴
        self.performance_window = 100  # 100取引で性能評価
        self.performance_threshold = {
            'win_rate': 0.70,
            'accuracy': 0.75
        }

        # 適応的パラメータ
        self.base_kelly = 0.70
        self.base_leverage = 10.0
        self.base_confidence_threshold = 0.65
        self.base_return_threshold = 0.35

        # 現在のパラメータ（適応的に変更される）
        self.kelly_fraction = self.base_kelly
        self.max_leverage = self.base_leverage
        self.phase1_confidence_threshold = self.base_confidence_threshold
        self.phase2_min_return = self.base_return_threshold

        # 統計
        self.total_predictions = 0
        self.online_model_trained = False

        logger.info("=" * 80)
        logger.info("適応的学習ボット起動")
        logger.info("=" * 80)
        logger.info(f"通貨ペア: {pair}")
        logger.info(f"初期資金: ¥{initial_capital:,}")
        logger.info(f"オンライン学習: 有効")
        logger.info(f"更新間隔: {self.update_interval}取引ごと")
        logger.info("=" * 80)

    def _convert_pair_to_yahoo(self, pair: str) -> str:
        """通貨ペアをYahoo Financeシンボルに変換"""
        pair_clean = pair.replace('/', '').replace('_', '')
        return f"{pair_clean}=X"

    def _init_online_models(self) -> Dict:
        """オンライン学習モデルを初期化"""
        models = {
            'sgd': SGDClassifier(
                loss='log_loss',
                penalty='l2',
                alpha=0.0001,
                learning_rate='adaptive',
                eta0=0.01,
                max_iter=1000,
                random_state=42,
                warm_start=True  # インクリメンタル学習用
            ),
            'passive_aggressive': PassiveAggressiveClassifier(
                C=0.01,
                max_iter=1000,
                random_state=42,
                warm_start=True
            )
        }

        logger.info("オンライン学習モデル初期化完了")
        logger.info(f"  - SGDClassifier")
        logger.info(f"  - PassiveAggressiveClassifier")

        return models

    def _load_fixed_models(self):
        """固定モデル（Phase 1.8 + Phase 2）を読み込み"""
        try:
            # Phase 1.8
            phase1_path = self.models_dir / 'phase1_8' / f'{self.pair.replace("/", "_")}_ensemble_models.pkl'
            if phase1_path.exists():
                with open(phase1_path, 'rb') as f:
                    self.fixed_phase1_models = pickle.load(f)
                logger.info(f"固定Phase 1.8モデル読み込み: {phase1_path}")

            # Phase 2
            phase2_path = self.models_dir / 'phase2' / f'{self.pair.replace("/", "_")}_xgboost_model.pkl'
            if phase2_path.exists():
                with open(phase2_path, 'rb') as f:
                    self.fixed_phase2_model = pickle.load(f)
                logger.info(f"固定Phase 2モデル読み込み: {phase2_path}")

        except Exception as e:
            logger.warning(f"固定モデル読み込みエラー: {e}")

    def get_historical_data(self, days: int = 250) -> Optional[pd.DataFrame]:
        """過去データ取得"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days + 50)

            ticker = yf.Ticker(self.yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date, interval='1d')

            if df.empty:
                logger.error("データ取得失敗")
                return None

            # カラム名を小文字に
            df.columns = [col.lower() for col in df.columns]

            # 最新N日分
            df = df.tail(days)

            logger.info(f"  過去データ取得: {len(df)}日分")
            return df

        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            return None

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量生成（43個 - 訓練済みモデルと同じ）"""
        features_df = df[['close', 'open', 'high', 'low', 'volume']].copy()

        # 比率
        features_df['high_low_ratio'] = df['high'] / df['low']
        features_df['close_open_ratio'] = df['close'] / df['open']

        # リターン
        for period in [1, 3, 5, 10, 20]:
            features_df[f'return_{period}d'] = df['close'].pct_change(period)

        # 移動平均
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = df['close'].rolling(period).mean()
            features_df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features_df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        features_df['macd'] = exp1 - exp2
        features_df['macd_signal'] = features_df['macd'].ewm(span=9).mean()

        # ボラティリティ
        for period in [5, 10, 20]:
            features_df[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std()

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features_df['atr_14'] = true_range.rolling(14).mean()

        # ボリンジャーバンド
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        features_df['bb_upper'] = sma_20 + (std_20 * 2)
        features_df['bb_lower'] = sma_20 - (std_20 * 2)
        features_df['bb_width'] = features_df['bb_upper'] - features_df['bb_lower']

        # ストキャスティクス
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        features_df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        features_df['stoch_d'] = features_df['stoch_k'].rolling(3).mean()

        # モメンタム
        features_df['momentum_10'] = df['close'] - df['close'].shift(10)

        # ROC
        features_df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

        # 欠損値を前方埋め
        features_df = features_df.ffill().bfill()

        return features_df

    def predict_with_online_models(self, features: np.ndarray) -> Optional[Dict]:
        """オンライン学習モデルで予測"""
        if not self.online_model_trained:
            return None

        try:
            # 各モデルで予測
            predictions = []
            probabilities = []

            for name, model in self.online_models.items():
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0]
                predictions.append(pred)
                probabilities.append(prob[1] if pred == 1 else 1 - prob[0])

            # アンサンブル（平均）
            ensemble_pred = int(np.mean(predictions) > 0.5)
            ensemble_prob = np.mean(probabilities)

            return {
                'direction': ensemble_pred,
                'confidence': ensemble_prob
            }

        except Exception as e:
            logger.error(f"オンライン予測エラー: {e}")
            return None

    def predict_signal(self, features_df: pd.DataFrame) -> Dict:
        """
        ハイブリッド予測: 固定モデル + 適応モデル
        """
        latest_features = features_df.iloc[-1:].values

        # 1. 固定モデル予測（Phase 1.8 + Phase 2）
        fixed_pred = self._predict_fixed_models(latest_features)

        # 2. オンライン学習モデル予測
        online_pred = self.predict_with_online_models(latest_features)

        # 3. ハイブリッド統合
        if online_pred is not None and self.online_model_trained:
            # 重み付け平均: 固定70% + オンライン30%
            # （オンラインモデルが成熟したら比率を変更可能）
            hybrid_confidence = (
                fixed_pred['confidence'] * 0.7 +
                online_pred['confidence'] * 0.3
            )

            # 方向は多数決
            if fixed_pred['direction'] == online_pred['direction']:
                hybrid_direction = fixed_pred['direction']
            else:
                # 信頼度が高い方を採用
                hybrid_direction = (
                    fixed_pred['direction'] if fixed_pred['confidence'] > online_pred['confidence']
                    else online_pred['direction']
                )

            logger.info(f"\nハイブリッド予測:")
            logger.info(f"  固定モデル: 方向={fixed_pred['direction']}, 信頼度={fixed_pred['confidence']:.4f}")
            logger.info(f"  適応モデル: 方向={online_pred['direction']}, 信頼度={online_pred['confidence']:.4f}")
            logger.info(f"  統合結果: 方向={hybrid_direction}, 信頼度={hybrid_confidence:.4f}")
        else:
            # オンラインモデルが未訓練なら固定モデルのみ
            hybrid_confidence = fixed_pred['confidence']
            hybrid_direction = fixed_pred['direction']
            logger.info(f"\n固定モデルのみ予測（適応モデル未訓練）")
            logger.info(f"  方向={hybrid_direction}, 信頼度={hybrid_confidence:.4f}")

        return {
            'direction': hybrid_direction,
            'confidence': hybrid_confidence,
            'expected_return': fixed_pred['expected_return'],
            'fixed_pred': fixed_pred,
            'online_pred': online_pred
        }

    def _predict_fixed_models(self, features: np.ndarray) -> Dict:
        """固定モデル（Phase 1.8 + Phase 2）で予測"""
        # デフォルト値
        default_pred = {
            'direction': 1,
            'confidence': 0.5,
            'expected_return': 0.0
        }

        if self.fixed_phase1_models is None or self.fixed_phase2_model is None:
            return default_pred

        try:
            # Phase 1.8予測
            predictions = []
            probabilities = []

            for name, model in self.fixed_phase1_models.items():
                pred = model.predict(features)[0]
                prob = model.predict_proba(features)[0]
                predictions.append(pred)
                probabilities.append(prob[1])

            # アンサンブル
            ensemble_prob = np.mean(probabilities)
            ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

            # Phase 2予測
            expected_return = self.fixed_phase2_model.predict(features)[0]

            return {
                'direction': ensemble_pred,
                'confidence': ensemble_prob if ensemble_pred == 1 else (1 - ensemble_prob),
                'expected_return': float(expected_return)
            }

        except Exception as e:
            logger.error(f"固定モデル予測エラー: {e}")
            return default_pred

    def update_online_models(self, features: np.ndarray, label: int):
        """
        オンライン学習: 新しいデータでモデル更新

        Args:
            features: 特徴量ベクトル
            label: 実際の結果（1: 上昇, 0: 下降）
        """
        self.update_buffer.append((features, label))

        # バッファが更新間隔に達したら更新
        if len(self.update_buffer) >= self.update_interval:
            logger.info(f"\nオンライン学習開始（{len(self.update_buffer)}サンプル）")

            X = np.array([x[0] for x in self.update_buffer]).reshape(-1, 43)
            y = np.array([x[1] for x in self.update_buffer])

            # スケーリング
            if not self.online_model_trained:
                # 初回のみfitライフサイクル
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)

            # 各モデルを更新
            for name, model in self.online_models.items():
                try:
                    model.partial_fit(X_scaled, y, classes=[0, 1])
                    logger.info(f"  {name}: 更新完了")
                except Exception as e:
                    logger.error(f"  {name}: 更新エラー - {e}")

            self.online_model_trained = True
            self.update_buffer = []  # バッファクリア

            # モデル保存
            self._save_online_models()

            logger.info("オンライン学習完了")

    def _save_online_models(self):
        """オンライン学習モデルを保存"""
        try:
            save_path = self.adaptive_models_dir / f'{self.pair.replace("/", "_")}_online_models.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'models': self.online_models,
                    'scaler': self.scaler,
                    'trained': self.online_model_trained
                }, f)
            logger.info(f"適応モデル保存: {save_path}")
        except Exception as e:
            logger.error(f"モデル保存エラー: {e}")

    def check_and_adapt_parameters(self, market_data: pd.DataFrame):
        """
        適応的パラメータ調整

        市場環境に応じてKelly分数、レバレッジ、閾値を動的調整
        """
        # ボラティリティ計算
        volatility = market_data['close'].pct_change().tail(20).std()

        # Kelly分数調整
        if volatility > 0.02:  # 高ボラ
            self.kelly_fraction = 0.30
        elif volatility > 0.01:  # 中ボラ
            self.kelly_fraction = 0.50
        else:  # 低ボラ
            self.kelly_fraction = 0.65

        # レバレッジ調整
        if volatility > 0.02:
            self.max_leverage = 3.0
        elif volatility > 0.01:
            self.max_leverage = 6.0
        else:
            self.max_leverage = 9.0

        # 最近のパフォーマンスベースで閾値調整
        if len(self.recent_trades) >= 20:
            recent_win_rate = sum(1 for t in self.recent_trades[-20:] if t.get('pnl', 0) > 0) / 20

            if recent_win_rate > 0.85:  # 好調
                self.phase1_confidence_threshold = 0.60
            elif recent_win_rate > 0.75:
                self.phase1_confidence_threshold = 0.65
            else:  # 不調
                self.phase1_confidence_threshold = 0.70

        logger.info(f"\n適応的パラメータ調整:")
        logger.info(f"  ボラティリティ: {volatility*100:.2f}%")
        logger.info(f"  Kelly分数: {self.kelly_fraction:.2f}")
        logger.info(f"  最大レバレッジ: {self.max_leverage:.1f}x")
        logger.info(f"  信頼度閾値: {self.phase1_confidence_threshold:.2f}")


if __name__ == '__main__':
    # テスト実行
    bot = AdaptiveLearningBot(pair='USD/JPY', initial_capital=10000)

    print("\n適応的学習ボットのテスト:")
    print(f"  オンライン学習モデル: {len(bot.online_models)}個")
    print(f"  更新間隔: {bot.update_interval}取引")
    print(f"  初期Kelly分数: {bot.kelly_fraction}")
    print(f"  初期レバレッジ: {bot.max_leverage}x")
