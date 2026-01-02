"""
Phase 2 予測エンジン

リアルタイム予測システム
Phase 1.8（93.64%精度）モデルをベースに実取引用に最適化
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import pickle
import warnings
warnings.filterwarnings('ignore')

# Phase 2コンポーネント
from src.phase2.realtime_pipeline import RealtimeDataPipeline
from src.phase2.database_manager import DatabaseManager


class PredictionEngine:
    """
    リアルタイム予測エンジン

    Phase 1.8の93.64%精度モデルを実取引用に適用

    予測フロー:
    1. リアルタイムデータ取得（RealtimeDataPipeline）
    2. 特徴量生成（1200+個）
    3. モデル予測（複数モデルのアンサンブル）
    4. 信頼度フィルタリング（閾値0.65）
    5. シグナル生成
    """

    def __init__(
        self,
        model_dir: Path = Path("models/phase1_8"),
        instruments: List[str] = None
    ):
        """
        Parameters:
        -----------
        model_dir : Path
            Phase 1.8モデルディレクトリ
        instruments : List[str], optional
            対象通貨ペア
        """
        self.model_dir = Path(model_dir)
        self.instruments = instruments or ['USD_JPY']

        # リアルタイムパイプライン初期化
        self.pipeline = RealtimeDataPipeline(instruments=self.instruments)

        # データベース
        self.db = DatabaseManager()

        # モデル読み込み
        self.models = {}
        self.feature_names = None
        self._load_models()

        # 予測設定（Phase 1.8と同じ）
        self.confidence_threshold = 0.65  # 信頼度閾値
        self.price_threshold = 0.5  # 価格変動閾値（%）

        logger.info("予測エンジン初期化完了")
        logger.info(f"  モデル数: {len(self.models)}")
        logger.info(f"  対象通貨ペア: {len(self.instruments)}")

    def _load_models(self):
        """
        Phase 1.8モデル読み込み

        Phase 1.8のアンサンブルモデル:
        - GradientBoostingClassifier
        - RandomForestClassifier
        - XGBClassifier
        - LGBMClassifier
        - CatBoostClassifier
        """
        model_files = [
            'gradient_boosting.pkl',
            'random_forest.pkl',
            'xgboost.pkl',
            'lightgbm.pkl',
            'catboost.pkl'
        ]

        for model_file in model_files:
            model_path = self.model_dir / model_file

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        model_name = model_file.replace('.pkl', '')
                        self.models[model_name] = model
                        logger.info(f"  ✓ {model_name} 読み込み完了")
                except Exception as e:
                    logger.warning(f"モデル読み込みエラー ({model_file}): {e}")
            else:
                logger.warning(f"モデルファイルが見つかりません: {model_path}")

        # 特徴量名読み込み
        feature_file = self.model_dir / 'feature_names.pkl'
        if feature_file.exists():
            with open(feature_file, 'rb') as f:
                self.feature_names = pickle.load(f)
                logger.info(f"  ✓ 特徴量名読み込み: {len(self.feature_names)}個")

        if not self.models:
            logger.warning("⚠️  モデルが読み込まれていません。ダミー予測を使用します。")

    def predict(
        self,
        instrument: str,
        timestamp: datetime = None
    ) -> Dict:
        """
        リアルタイム予測

        Parameters:
        -----------
        instrument : str
            通貨ペア
        timestamp : datetime, optional
            予測時刻（デフォルト: 現在時刻）

        Returns:
        --------
        Dict
            {
                'direction': 方向（1=上昇, -1=下降, 0=中立）,
                'probability': 確率（0-1）,
                'confidence': 信頼度（0-1）,
                'should_trade': 取引すべきか,
                'current_price': 現在価格,
                'timestamp': 予測時刻,
                'models': {モデル名: 予測}
            }
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"\n予測開始: {instrument} @ {timestamp}")

        # 1. リアルタイムデータ・特徴量取得
        signal_data = self.pipeline.get_current_signal(instrument)
        features = signal_data['features']
        current_price = signal_data['price']['mid']

        # 2. 特徴量を正しい順序に並べる
        if self.feature_names:
            # 必要な特徴量のみ抽出（Phase 1.8で使用した60個）
            feature_vector = []
            for feat_name in self.feature_names:
                feature_vector.append(features.get(feat_name, 0))

            X = np.array(feature_vector).reshape(1, -1)
        else:
            # 特徴量名がない場合はダミー
            logger.warning("特徴量名が見つかりません。ダミーデータを使用。")
            X = np.random.randn(1, 60)

        # 3. 各モデルで予測
        model_predictions = {}

        if self.models:
            for model_name, model in self.models.items():
                try:
                    # 予測確率取得
                    proba = model.predict_proba(X)[0]

                    # クラス1（上昇）の確率
                    prob_up = proba[1] if len(proba) > 1 else 0.5

                    model_predictions[model_name] = {
                        'probability': prob_up,
                        'direction': 1 if prob_up > 0.5 else -1
                    }

                except Exception as e:
                    logger.warning(f"モデル予測エラー ({model_name}): {e}")
        else:
            # ダミー予測（モデルがない場合）
            logger.warning("モデルがないため、ダミー予測を使用")
            model_predictions['dummy'] = {
                'probability': 0.5,
                'direction': 0
            }

        # 4. アンサンブル予測（加重平均）
        # Phase 1.8では各モデルに重みを設定
        model_weights = {
            'gradient_boosting': 0.25,
            'random_forest': 0.20,
            'xgboost': 0.25,
            'lightgbm': 0.15,
            'catboost': 0.15
        }

        weighted_prob = 0
        total_weight = 0

        for model_name, pred in model_predictions.items():
            weight = model_weights.get(model_name, 1.0 / len(model_predictions))
            weighted_prob += pred['probability'] * weight
            total_weight += weight

        ensemble_probability = weighted_prob / total_weight if total_weight > 0 else 0.5

        # 5. 方向判定
        if ensemble_probability >= 0.5:
            direction = 1  # 上昇
            confidence = ensemble_probability
        else:
            direction = -1  # 下降
            confidence = 1 - ensemble_probability

        # 6. 取引判定（Phase 1.8の信頼度フィルタ）
        should_trade = confidence >= self.confidence_threshold

        # 7. 予測結果
        prediction = {
            'direction': direction,
            'probability': ensemble_probability,
            'confidence': confidence,
            'should_trade': should_trade,
            'current_price': current_price,
            'timestamp': timestamp,
            'instrument': instrument,
            'models': model_predictions,
            'features': features
        }

        # 8. データベースに保存
        self.db.save_prediction(
            timestamp=timestamp,
            instrument=instrument,
            model_name='phase1_8_ensemble',
            prediction=prediction
        )

        logger.info(f"  ✓ 予測完了: {instrument}")
        logger.info(f"    方向: {'上昇' if direction == 1 else '下降'}")
        logger.info(f"    確率: {ensemble_probability:.4f}")
        logger.info(f"    信頼度: {confidence:.4f}")
        logger.info(f"    取引可否: {'YES' if should_trade else 'NO'}")

        return prediction

    def predict_multiple(
        self,
        instruments: List[str] = None,
        timestamp: datetime = None
    ) -> Dict[str, Dict]:
        """
        複数通貨ペアの予測

        Parameters:
        -----------
        instruments : List[str], optional
            通貨ペアリスト（デフォルト: self.instruments）
        timestamp : datetime, optional
            予測時刻

        Returns:
        --------
        Dict[str, Dict]
            {通貨ペア: 予測結果}
        """
        if instruments is None:
            instruments = self.instruments

        predictions = {}

        for instrument in instruments:
            try:
                pred = self.predict(instrument, timestamp)
                predictions[instrument] = pred
            except Exception as e:
                logger.error(f"予測エラー ({instrument}): {e}")

        return predictions

    def get_tradeable_signals(
        self,
        instruments: List[str] = None
    ) -> List[Dict]:
        """
        取引可能シグナル取得

        信頼度閾値を超えたシグナルのみ返す

        Returns:
        --------
        List[Dict]
            取引可能なシグナルリスト
        """
        predictions = self.predict_multiple(instruments)

        tradeable = []

        for instrument, pred in predictions.items():
            if pred['should_trade']:
                tradeable.append({
                    'instrument': instrument,
                    'direction': pred['direction'],
                    'confidence': pred['confidence'],
                    'probability': pred['probability'],
                    'price': pred['current_price'],
                    'timestamp': pred['timestamp']
                })

        return tradeable

    def evaluate_prediction(
        self,
        prediction_id: int,
        actual_price: float
    ):
        """
        予測の答え合わせ

        Parameters:
        -----------
        prediction_id : int
            予測ID
        actual_price : float
            実際の価格
        """
        # 予測データ取得
        cursor = self.db.conn.execute("""
            SELECT current_price, prediction_direction
            FROM predictions
            WHERE id = ?
        """, (prediction_id,))

        result = cursor.fetchone()

        if result:
            initial_price, predicted_direction = result

            # 実際のリターン
            actual_return = ((actual_price / initial_price) - 1) * 100

            # 実際の方向
            actual_direction = 1 if actual_return > self.price_threshold else (-1 if actual_return < -self.price_threshold else 0)

            # データベース更新
            self.db.update_prediction_result(
                prediction_id=prediction_id,
                actual_direction=actual_direction,
                actual_return=actual_return
            )

            logger.info(f"予測評価完了: ID={prediction_id}, 実際={actual_direction}, 予測={predicted_direction}")


if __name__ == "__main__":
    # テスト
    logger.info("予測エンジン テスト開始\n")

    # 予測エンジン初期化
    engine = PredictionEngine(instruments=['USD_JPY'])

    # 単一予測
    logger.info("=== 単一予測テスト ===")
    prediction = engine.predict('USD_JPY')

    # 複数通貨予測
    logger.info("\n=== 複数通貨予測テスト ===")
    predictions = engine.predict_multiple(['USD_JPY', 'EUR_USD'])

    # 取引可能シグナル
    logger.info("\n=== 取引可能シグナル取得 ===")
    tradeable = engine.get_tradeable_signals()
    logger.info(f"取引可能シグナル: {len(tradeable)}個")
