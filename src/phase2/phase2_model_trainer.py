"""
Phase 2 モデル訓練システム

Phase 1との決定的な違い:
- Phase 1: 方向予測（上昇 or 下降） → 精度93.64%達成
- Phase 2: 収益予測（どれくらい利益が出るか） → Sharpe Ratio最大化

訓練目標:
- 損失関数: Sharpe Ratio最大化
- 評価指標: 実現PnL、リターン、勝率
- ポジションサイジング統合
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import pickle
import warnings
warnings.filterwarnings('ignore')

# 機械学習
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# データソース（Phase 1.8と同じ）
from src.data_sources.yahoo_finance import YahooFinanceData
from src.data_sources.economic_indicators import EconomicIndicators

# 評価指標
from scipy import stats


class Phase2ModelTrainer:
    """
    Phase 2 収益最適化モデル訓練システム

    Phase 1との違い:
    1. ラベル: 方向 → **実際のリターン（%）**
    2. モデル: 分類器 → **回帰器**
    3. 損失: 精度 → **Sharpe Ratio**
    4. 評価: 的中率 → **実現PnL**

    訓練データ:
    - 10年分データ（Phase 1.8と同じ）
    - 特徴量: 1200+個
    - ラベル: 翌日リターン（%）
    """

    def __init__(
        self,
        instruments: List[str] = None,
        lookback_days: int = 2500,
        output_dir: Path = Path("models/phase2")
    ):
        """
        Parameters:
        -----------
        instruments : List[str], optional
            対象通貨ペア
        lookback_days : int
            訓練データ期間（日数）
        output_dir : Path
            モデル保存先
        """
        self.instruments = instruments or ['USD_JPY']
        self.lookback_days = lookback_days
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # データソース
        self.yahoo = YahooFinanceData()
        self.economic = EconomicIndicators()

        # モデル
        self.models = {}
        self.scaler = StandardScaler()

        logger.info("Phase 2 モデルトレーナー初期化完了")
        logger.info(f"  対象: {self.instruments}")
        logger.info(f"  訓練期間: {lookback_days}日")

    def collect_training_data(
        self,
        instrument: str
    ) -> pd.DataFrame:
        """
        訓練データ収集

        Phase 1.8と同じデータソースを使用
        ただし、ラベルは「実際のリターン」

        Returns:
        --------
        pd.DataFrame
            特徴量 + リターンラベル
        """
        logger.info(f"\n訓練データ収集開始: {instrument}")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        # 1. 価格データ取得（Yahoo Finance）
        # USD_JPY → USD/JPY形式に変換
        pair = instrument.replace('_', '/')
        price_data = self.yahoo.get_forex_data(
            pair=pair,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if price_data.empty:
            logger.error(f"価格データ取得失敗: {instrument}")
            return pd.DataFrame()

        logger.info(f"  ✓ 価格データ: {len(price_data)}日分")

        # 2. リターン計算（これがPhase 2のラベル）
        price_data['return_1d'] = price_data['close'].pct_change(1) * 100  # 翌日リターン（%）
        price_data['return_5d'] = price_data['close'].pct_change(5) * 100  # 5日リターン
        price_data['return_20d'] = price_data['close'].pct_change(20) * 100  # 20日リターン

        # 3. テクニカル指標特徴量
        features_df = self._generate_technical_features(price_data)

        # 4. 経済指標特徴量
        for date in price_data.index:
            try:
                econ_features = self.economic.generate_features(instrument, date)
                for feat_name, feat_value in econ_features.items():
                    if feat_name not in features_df.columns:
                        features_df[feat_name] = np.nan
                    features_df.loc[date, feat_name] = feat_value
            except:
                pass

        # 5. ラベル追加（Phase 2の重要ポイント）
        features_df['label_return_1d'] = price_data['return_1d'].shift(-1)  # 翌日リターン
        features_df['label_return_5d'] = price_data['return_5d'].shift(-5)  # 5日後リターン

        # NaN除去
        features_df = features_df.dropna()

        logger.info(f"  ✓ 特徴量生成完了: {features_df.shape}")
        logger.info(f"     特徴量数: {features_df.shape[1] - 2}個")  # ラベル2個を除く
        logger.info(f"     サンプル数: {len(features_df)}")

        return features_df

    def _generate_technical_features(
        self,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        テクニカル指標特徴量生成

        Phase 1.8と同じ特徴量を使用
        """
        df = price_data.copy()
        features = pd.DataFrame(index=df.index)

        # 基本価格情報
        features['close'] = df['close']
        features['high'] = df['high']
        features['low'] = df['low']
        features['volume'] = df['volume']

        # リターン
        for period in [1, 5, 10, 20, 60]:
            features[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # SMA
        for period in [5, 10, 20, 50, 100, 200]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = sma
            features[f'price_vs_sma_{period}'] = ((df['close'] / sma) - 1) * 100

        # EMA
        for period in [12, 26, 50]:
            ema = df['close'].ewm(span=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_vs_ema_{period}'] = ((df['close'] / ema) - 1) * 100

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal

        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        features['bb_upper'] = sma20 + (2 * std20)
        features['bb_middle'] = sma20
        features['bb_lower'] = sma20 - (2 * std20)
        features['bb_width'] = ((features['bb_upper'] - features['bb_lower']) / sma20)
        features['bb_position'] = ((df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']))

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features['atr_14'] = tr.rolling(14).mean()
        features['atr_ratio'] = (features['atr_14'] / df['close']) * 100

        # ボラティリティ
        for period in [5, 10, 20]:
            features[f'volatility_{period}d'] = df['close'].pct_change().rolling(period).std() * 100

        return features

    def train_models(
        self,
        instrument: str,
        target_label: str = 'label_return_1d'
    ) -> Dict:
        """
        Phase 2モデル訓練

        Parameters:
        -----------
        instrument : str
            通貨ペア
        target_label : str
            目標ラベル（'label_return_1d' or 'label_return_5d'）

        Returns:
        --------
        Dict
            訓練結果
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Phase 2 モデル訓練開始: {instrument}")
        logger.info(f"{'='*80}")

        # 1. データ収集
        data = self.collect_training_data(instrument)

        if data.empty:
            logger.error("訓練データが空です")
            return {}

        # 2. 特徴量とラベル分離
        feature_cols = [col for col in data.columns if not col.startswith('label_')]
        X = data[feature_cols].values
        y = data[target_label].values

        logger.info(f"\n特徴量: {X.shape}")
        logger.info(f"ラベル（リターン）: {y.shape}")
        logger.info(f"  平均リターン: {y.mean():.4f}%")
        logger.info(f"  リターン標準偏差: {y.std():.4f}%")
        logger.info(f"  Sharpe (年率): {(y.mean() / y.std() * np.sqrt(252)):.2f}")

        # 3. 訓練/検証/テスト分割（時系列考慮）
        train_size = int(len(X) * 0.7)
        val_size = int(len(X) * 0.15)

        X_train = X[:train_size]
        y_train = y[:train_size]

        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]

        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]

        logger.info(f"\n訓練: {len(X_train)}, 検証: {len(X_val)}, テスト: {len(X_test)}")

        # 4. スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 5. モデル訓練（回帰器）
        models = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                verbose=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=200,
                learning_rate=0.05,
                depth=5,
                random_state=42,
                verbose=False
            )
        }

        results = {}

        for model_name, model in models.items():
            logger.info(f"\n訓練中: {model_name}...")

            # 訓練
            model.fit(X_train_scaled, y_train)

            # 予測
            y_pred_train = model.predict(X_train_scaled)
            y_pred_val = model.predict(X_val_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Phase 2評価指標（収益重視）
            train_metrics = self._calculate_profit_metrics(y_train, y_pred_train)
            val_metrics = self._calculate_profit_metrics(y_val, y_pred_val)
            test_metrics = self._calculate_profit_metrics(y_test, y_pred_test)

            results[model_name] = {
                'model': model,
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics
            }

            logger.info(f"  ✓ テストSharpe: {test_metrics['sharpe']:.2f}")
            logger.info(f"  ✓ テスト累積リターン: {test_metrics['cumulative_return']:.2f}%")

        # 6. ベストモデル選択（Sharpe Ratio基準）
        best_model_name = max(
            results.keys(),
            key=lambda x: results[x]['val']['sharpe']
        )

        logger.info(f"\n{'='*80}")
        logger.info(f"ベストモデル: {best_model_name}")
        logger.info(f"  検証Sharpe: {results[best_model_name]['val']['sharpe']:.2f}")
        logger.info(f"  テストSharpe: {results[best_model_name]['test']['sharpe']:.2f}")
        logger.info(f"{'='*80}")

        # 7. モデル保存
        self._save_models(instrument, results, feature_cols)

        return results

    def _calculate_profit_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Phase 2評価指標（収益重視）

        Returns:
        --------
        Dict
            {
                'sharpe': Sharpe Ratio,
                'cumulative_return': 累積リターン（%）,
                'win_rate': 勝率,
                'avg_win': 平均勝ちトレード,
                'avg_loss': 平均負けトレード,
                'profit_factor': プロフィットファクター
            }
        """
        # 予測に基づいてトレード
        # 予測が正の場合は買い、負の場合は売り
        positions = np.sign(y_pred)  # 1 (買い) or -1 (売り)
        returns = positions * y_true  # 実際のリターン

        # Sharpe Ratio（年率換算）
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # 累積リターン
        cumulative_return = returns.sum()

        # 勝率
        wins = returns > 0
        win_rate = wins.sum() / len(returns) * 100 if len(returns) > 0 else 0

        # 平均勝ち/負け
        avg_win = returns[wins].mean() if wins.sum() > 0 else 0
        avg_loss = returns[~wins].mean() if (~wins).sum() > 0 else 0

        # プロフィットファクター
        total_win = returns[wins].sum() if wins.sum() > 0 else 0
        total_loss = abs(returns[~wins].sum()) if (~wins).sum() > 0 else 1
        profit_factor = total_win / total_loss if total_loss > 0 else 0

        return {
            'sharpe': sharpe,
            'cumulative_return': cumulative_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(returns),
            'winning_trades': wins.sum(),
            'losing_trades': (~wins).sum()
        }

    def _save_models(
        self,
        instrument: str,
        results: Dict,
        feature_names: List[str]
    ):
        """モデル保存"""
        instrument_dir = self.output_dir / instrument
        instrument_dir.mkdir(parents=True, exist_ok=True)

        # 各モデル保存
        for model_name, result in results.items():
            model_file = instrument_dir / f"{model_name}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(result['model'], f)
            logger.info(f"  ✓ {model_name} 保存: {model_file}")

        # スケーラー保存
        scaler_file = instrument_dir / "scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)

        # 特徴量名保存
        features_file = instrument_dir / "feature_names.pkl"
        with open(features_file, 'wb') as f:
            pickle.dump(feature_names, f)

        # メタデータ保存
        metadata = {
            'instrument': instrument,
            'training_date': datetime.now().isoformat(),
            'num_features': len(feature_names),
            'models': list(results.keys()),
            'best_model': max(results.keys(), key=lambda x: results[x]['val']['sharpe']),
            'test_results': {
                model_name: {
                    'sharpe': result['test']['sharpe'],
                    'cumulative_return': result['test']['cumulative_return'],
                    'win_rate': result['test']['win_rate']
                }
                for model_name, result in results.items()
            }
        }

        metadata_file = instrument_dir / "metadata.json"
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"\n✅ 全モデル保存完了: {instrument_dir}")


if __name__ == "__main__":
    # テスト実行
    logger.info("Phase 2 モデルトレーナー テスト\n")

    trainer = Phase2ModelTrainer(
        instruments=['USD_JPY'],
        lookback_days=2500  # 10年分
    )

    # モデル訓練
    results = trainer.train_models('USD_JPY')

    # 結果表示
    logger.info("\n" + "="*80)
    logger.info("訓練完了！")
    logger.info("="*80)

    for model_name, result in results.items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  テストSharpe: {result['test']['sharpe']:.2f}")
        logger.info(f"  累積リターン: {result['test']['cumulative_return']:.2f}%")
        logger.info(f"  勝率: {result['test']['win_rate']:.2f}%")
        logger.info(f"  プロフィットファクター: {result['test']['profit_factor']:.2f}")
