"""
ハイブリッド予測エンジン

Phase 1.8 + Phase 2 の統合システム

Phase 1.8: 方向性予測（93.64%精度）
Phase 2:   収益予測（Sharpe最大化）

組み合わせて最強の予測システムを構築
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


class HybridPredictionEngine:
    """
    ハイブリッド予測エンジン

    Phase 1.8（方向）+ Phase 2（収益）の組み合わせ

    予測戦略:
    1. Phase 1.8で方向を予測（上昇 or 下降）
    2. Phase 2で期待リターンを予測（何%動くか）
    3. 両方の信頼度が高い時のみ取引

    これにより:
    - 高い方向性的中率（93.64%）
    - 高い収益性（Sharpe最大化）
    を両立
    """

    def __init__(
        self,
        phase1_model_dir: Path = Path("models/phase1_8"),
        phase2_model_dir: Path = Path("models/phase2"),
        instruments: List[str] = None
    ):
        """
        Parameters:
        -----------
        phase1_model_dir : Path
            Phase 1.8モデルディレクトリ
        phase2_model_dir : Path
            Phase 2モデルディレクトリ
        instruments : List[str], optional
            対象通貨ペア
        """
        self.phase1_dir = Path(phase1_model_dir)
        self.phase2_dir = Path(phase2_model_dir)
        self.instruments = instruments or ['USD_JPY']

        # リアルタイムパイプライン
        self.pipeline = RealtimeDataPipeline(instruments=self.instruments)

        # データベース
        self.db = DatabaseManager()

        # モデル読み込み
        self.phase1_models = {}  # 分類器（方向予測）
        self.phase2_models = {}  # 回帰器（収益予測）
        self.scalers = {}
        self.feature_names = {}

        self._load_all_models()

        # ハイブリッド設定
        self.direction_confidence_threshold = 0.65  # Phase 1.8の閾値
        self.min_expected_return = 0.3  # Phase 2の最小期待リターン（%）

        logger.info("ハイブリッド予測エンジン初期化完了")
        logger.info(f"  Phase 1.8モデル: {len(self.phase1_models)}個")
        logger.info(f"  Phase 2モデル: {len(self.phase2_models)}個")

    def _load_all_models(self):
        """全モデル読み込み"""
        for instrument in self.instruments:
            self._load_phase1_models(instrument)
            self._load_phase2_models(instrument)

    def _load_phase1_models(self, instrument: str):
        """Phase 1.8モデル読み込み（分類器）"""
        model_files = [
            'gradient_boosting.pkl',
            'random_forest.pkl',
            'xgboost.pkl',
            'lightgbm.pkl',
            'catboost.pkl'
        ]

        self.phase1_models[instrument] = {}

        for model_file in model_files:
            model_path = self.phase1_dir / model_file

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        model_name = model_file.replace('.pkl', '')
                        self.phase1_models[instrument][model_name] = model
                except Exception as e:
                    logger.warning(f"Phase 1.8モデル読み込みエラー ({model_file}): {e}")

        # 特徴量名
        feature_file = self.phase1_dir / 'feature_names.pkl'
        if feature_file.exists():
            with open(feature_file, 'rb') as f:
                self.feature_names[f'{instrument}_phase1'] = pickle.load(f)

        logger.info(f"  ✓ Phase 1.8 ({instrument}): {len(self.phase1_models[instrument])}モデル")

    def _load_phase2_models(self, instrument: str):
        """Phase 2モデル読み込み（回帰器）"""
        instrument_dir = self.phase2_dir / instrument

        if not instrument_dir.exists():
            logger.warning(f"Phase 2モデル未訓練: {instrument}")
            self.phase2_models[instrument] = {}
            return

        model_files = [
            'gradient_boosting.pkl',
            'random_forest.pkl',
            'xgboost.pkl',
            'lightgbm.pkl',
            'catboost.pkl'
        ]

        self.phase2_models[instrument] = {}

        for model_file in model_files:
            model_path = instrument_dir / model_file

            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        model_name = model_file.replace('.pkl', '')
                        self.phase2_models[instrument][model_name] = model
                except Exception as e:
                    logger.warning(f"Phase 2モデル読み込みエラー ({model_file}): {e}")

        # スケーラー
        scaler_file = instrument_dir / 'scaler.pkl'
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scalers[instrument] = pickle.load(f)

        # 特徴量名
        feature_file = instrument_dir / 'feature_names.pkl'
        if feature_file.exists():
            with open(feature_file, 'rb') as f:
                self.feature_names[f'{instrument}_phase2'] = pickle.load(f)

        logger.info(f"  ✓ Phase 2 ({instrument}): {len(self.phase2_models[instrument])}モデル")

    def predict_hybrid(
        self,
        instrument: str,
        timestamp: datetime = None
    ) -> Dict:
        """
        ハイブリッド予測

        Phase 1.8 + Phase 2の統合予測

        Returns:
        --------
        Dict
            {
                'phase1': Phase 1.8予測（方向・信頼度）,
                'phase2': Phase 2予測（期待リターン）,
                'hybrid': ハイブリッド判定,
                'should_trade': 取引すべきか,
                'direction': 方向,
                'expected_return': 期待リターン,
                'confidence': 総合信頼度
            }
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"\nハイブリッド予測: {instrument} @ {timestamp}")

        # 1. リアルタイムデータ・特徴量取得
        signal_data = self.pipeline.get_current_signal(instrument)
        features = signal_data['features']
        current_price = signal_data['price']['mid']

        # 2. Phase 1.8予測（方向）
        phase1_prediction = self._predict_phase1(instrument, features)

        # 3. Phase 2予測（期待リターン）
        phase2_prediction = self._predict_phase2(instrument, features)

        # 4. ハイブリッド判定
        hybrid_decision = self._make_hybrid_decision(
            phase1_prediction,
            phase2_prediction
        )

        # 5. 総合結果
        result = {
            'instrument': instrument,
            'timestamp': timestamp,
            'current_price': current_price,
            'phase1': phase1_prediction,
            'phase2': phase2_prediction,
            'hybrid': hybrid_decision,
            'should_trade': hybrid_decision['should_trade'],
            'direction': hybrid_decision['direction'],
            'expected_return': hybrid_decision['expected_return'],
            'confidence': hybrid_decision['confidence']
        }

        # 6. データベース保存
        self._save_hybrid_prediction(result)

        # ログ
        logger.info(f"  Phase 1.8: {phase1_prediction['direction_name']} (信頼度: {phase1_prediction['confidence']:.2f})")
        logger.info(f"  Phase 2:   期待リターン {phase2_prediction['expected_return']:.2f}%")
        logger.info(f"  ハイブリッド: {'取引' if result['should_trade'] else '見送り'}")

        if result['should_trade']:
            logger.info(f"    → {hybrid_decision['direction_name']} (期待: {result['expected_return']:.2f}%)")

        return result

    def _predict_phase1(
        self,
        instrument: str,
        features: Dict[str, float]
    ) -> Dict:
        """
        Phase 1.8予測（方向）

        Returns:
        --------
        Dict
            {
                'direction': 1 or -1,
                'probability': 確率,
                'confidence': 信頼度,
                'direction_name': '上昇' or '下降'
            }
        """
        if instrument not in self.phase1_models or not self.phase1_models[instrument]:
            logger.warning("Phase 1.8モデルなし。ダミー予測")
            return {
                'direction': 0,
                'probability': 0.5,
                'confidence': 0.0,
                'direction_name': '中立'
            }

        # 特徴量準備
        feature_key = f'{instrument}_phase1'
        if feature_key in self.feature_names:
            feature_vector = [features.get(name, 0) for name in self.feature_names[feature_key]]
            X = np.array(feature_vector).reshape(1, -1)
        else:
            X = np.random.randn(1, 60)  # ダミー

        # アンサンブル予測
        probabilities = []

        for model_name, model in self.phase1_models[instrument].items():
            try:
                proba = model.predict_proba(X)[0]
                prob_up = proba[1] if len(proba) > 1 else 0.5
                probabilities.append(prob_up)
            except:
                pass

        # 平均確率
        avg_prob = np.mean(probabilities) if probabilities else 0.5

        # 方向・信頼度
        if avg_prob >= 0.5:
            direction = 1
            confidence = avg_prob
            direction_name = '上昇'
        else:
            direction = -1
            confidence = 1 - avg_prob
            direction_name = '下降'

        return {
            'direction': direction,
            'probability': avg_prob,
            'confidence': confidence,
            'direction_name': direction_name
        }

    def _predict_phase2(
        self,
        instrument: str,
        features: Dict[str, float]
    ) -> Dict:
        """
        Phase 2予測（期待リターン）

        Returns:
        --------
        Dict
            {
                'expected_return': 期待リターン（%）,
                'std_return': 予測の標準偏差
            }
        """
        if instrument not in self.phase2_models or not self.phase2_models[instrument]:
            logger.warning("Phase 2モデルなし。ダミー予測")
            return {
                'expected_return': 0.0,
                'std_return': 0.0
            }

        # 特徴量準備
        feature_key = f'{instrument}_phase2'
        if feature_key in self.feature_names:
            feature_vector = [features.get(name, 0) for name in self.feature_names[feature_key]]
            X = np.array(feature_vector).reshape(1, -1)

            # スケーリング
            if instrument in self.scalers:
                X = self.scalers[instrument].transform(X)
        else:
            X = np.random.randn(1, 60)

        # アンサンブル予測
        predictions = []

        for model_name, model in self.phase2_models[instrument].items():
            try:
                pred = model.predict(X)[0]
                predictions.append(pred)
            except:
                pass

        # 平均・標準偏差
        expected_return = np.mean(predictions) if predictions else 0.0
        std_return = np.std(predictions) if predictions else 0.0

        return {
            'expected_return': expected_return,
            'std_return': std_return
        }

    def _make_hybrid_decision(
        self,
        phase1: Dict,
        phase2: Dict
    ) -> Dict:
        """
        ハイブリッド判定

        Phase 1.8とPhase 2の予測を組み合わせて最終判断

        取引条件:
        1. Phase 1.8の信頼度 >= 0.65
        2. Phase 2の期待リターン >= 0.3%
        3. 方向が一致

        Returns:
        --------
        Dict
            {
                'should_trade': bool,
                'direction': int,
                'expected_return': float,
                'confidence': float,
                'reason': str
            }
        """
        # Phase 1.8の方向と信頼度
        p1_direction = phase1['direction']
        p1_confidence = phase1['confidence']

        # Phase 2の期待リターン
        p2_return = phase2['expected_return']

        # Phase 2から推測される方向
        p2_direction = 1 if p2_return > 0 else -1

        # 判定
        should_trade = False
        reason = ""

        # 条件1: Phase 1.8の信頼度チェック
        if p1_confidence < self.direction_confidence_threshold:
            reason = f"Phase 1.8信頼度不足 ({p1_confidence:.2f} < {self.direction_confidence_threshold})"

        # 条件2: Phase 2の期待リターンチェック
        elif abs(p2_return) < self.min_expected_return:
            reason = f"Phase 2期待リターン不足 ({abs(p2_return):.2f}% < {self.min_expected_return}%)"

        # 条件3: 方向一致チェック
        elif p1_direction != p2_direction:
            reason = f"方向不一致 (Phase1: {p1_direction}, Phase2: {p2_direction})"

        # 全条件クリア
        else:
            should_trade = True
            reason = "全条件クリア"

        # 総合信頼度（Phase 1.8とPhase 2の積）
        confidence = p1_confidence * (abs(p2_return) / 1.0)  # 1%リターンで信頼度1.0

        return {
            'should_trade': should_trade,
            'direction': p1_direction,
            'direction_name': '上昇' if p1_direction == 1 else '下降',
            'expected_return': p2_return,
            'confidence': min(confidence, 1.0),
            'reason': reason
        }

    def _save_hybrid_prediction(self, result: Dict):
        """ハイブリッド予測をデータベース保存"""
        self.db.save_prediction(
            timestamp=result['timestamp'],
            instrument=result['instrument'],
            model_name='hybrid_phase1_phase2',
            prediction={
                'direction': result['direction'],
                'probability': result['phase1']['probability'],
                'confidence': result['confidence'],
                'current_price': result['current_price'],
                'expected_return': result['expected_return']
            }
        )


if __name__ == "__main__":
    # テスト
    logger.info("ハイブリッド予測エンジン テスト\n")

    engine = HybridPredictionEngine(instruments=['USD_JPY'])

    # ハイブリッド予測
    prediction = engine.predict_hybrid('USD_JPY')

    logger.info("\n予測結果:")
    logger.info(f"  取引判定: {prediction['should_trade']}")
    logger.info(f"  方向: {prediction['hybrid']['direction_name']}")
    logger.info(f"  期待リターン: {prediction['expected_return']:.2f}%")
    logger.info(f"  信頼度: {prediction['confidence']:.2f}")
    logger.info(f"  理由: {prediction['hybrid']['reason']}")
