"""
Phase 2 シグナル生成システム

ハイブリッド予測エンジンから取引シグナルを生成

シグナル生成条件（PHASE2_ROADMAPに基づく）:
1. エントリータイミング最適化
   - Phase 1.8信頼度 >= 0.65
   - Phase 2期待リターン >= 0.3%
   - 複数時間足一致（D/H4/H1/M15）
   - ATR適正範囲
   - 重要ニュース回避

2. ポジションサイジング
   - Kelly基準
   - 信頼度ベース調整
   - ポートフォリオ制約

3. エグジット戦略
   - テイクプロフィット: ATR × 4
   - ストップロス: ATR × 2
   - トレーリングストップ
"""

from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Phase 2コンポーネント
from src.phase2.hybrid_prediction_engine import HybridPredictionEngine
from src.phase2.realtime_pipeline import RealtimeDataPipeline
from src.phase2.database_manager import DatabaseManager


class SignalGenerator:
    """
    取引シグナル生成システム

    ハイブリッド予測 → フィルタリング → シグナル生成

    生成されるシグナル:
    - direction: 1（買い）or -1（売り）
    - entry_price: エントリー価格
    - stop_loss: ストップロス価格
    - take_profit: テイクプロフィット価格
    - position_size: ポジションサイズ（単位）
    - confidence: 信頼度
    """

    def __init__(
        self,
        instruments: List[str] = None,
        account_balance: float = 100000.0  # 初期口座残高
    ):
        """
        Parameters:
        -----------
        instruments : List[str], optional
            対象通貨ペア
        account_balance : float
            口座残高（ポジションサイジング用）
        """
        self.instruments = instruments or ['USD_JPY']
        self.account_balance = account_balance

        # ハイブリッド予測エンジン
        self.prediction_engine = HybridPredictionEngine(instruments=self.instruments)

        # リアルタイムパイプライン
        self.pipeline = RealtimeDataPipeline(instruments=self.instruments)

        # データベース
        self.db = DatabaseManager()

        # シグナル設定
        self.kelly_fraction = 0.25  # Kelly基準の25%（保守的）
        self.max_position_size_pct = 0.15  # 口座の15%まで
        self.tp_atr_multiplier = 4.0  # テイクプロフィット = ATR × 4
        self.sl_atr_multiplier = 2.0  # ストップロス = ATR × 2

        logger.info("シグナル生成システム初期化完了")
        logger.info(f"  対象通貨ペア: {len(self.instruments)}")
        logger.info(f"  口座残高: {self.account_balance:,.0f}")

    def generate_signal(
        self,
        instrument: str,
        timestamp: datetime = None
    ) -> Optional[Dict]:
        """
        取引シグナル生成

        Parameters:
        -----------
        instrument : str
            通貨ペア
        timestamp : datetime, optional
            タイムスタンプ

        Returns:
        --------
        Dict or None
            シグナル（取引すべき場合）、None（見送り）
        """
        if timestamp is None:
            timestamp = datetime.now()

        logger.info(f"\n{'='*80}")
        logger.info(f"シグナル生成: {instrument} @ {timestamp}")
        logger.info(f"{'='*80}")

        # 1. ハイブリッド予測
        prediction = self.prediction_engine.predict_hybrid(instrument, timestamp)

        # 取引しない場合
        if not prediction['should_trade']:
            logger.info(f"  → 見送り: {prediction['hybrid']['reason']}")
            return None

        # 2. 追加フィルタリング
        if not self._additional_filters(instrument, prediction):
            logger.info("  → 見送り: 追加フィルタ不合格")
            return None

        # 3. エントリー価格・ATR取得
        current_price = prediction['current_price']
        atr = self._get_atr(instrument)

        if atr is None or atr == 0:
            logger.warning("  → 見送り: ATR取得失敗")
            return None

        # 4. ストップロス・テイクプロフィット計算
        direction = prediction['direction']

        if direction == 1:  # 買い
            stop_loss = current_price - (atr * self.sl_atr_multiplier)
            take_profit = current_price + (atr * self.tp_atr_multiplier)
        else:  # 売り
            stop_loss = current_price + (atr * self.sl_atr_multiplier)
            take_profit = current_price - (atr * self.tp_atr_multiplier)

        # 5. ポジションサイジング
        position_size = self._calculate_position_size(
            instrument=instrument,
            entry_price=current_price,
            stop_loss=stop_loss,
            confidence=prediction['confidence'],
            expected_return=prediction['expected_return']
        )

        # 6. シグナル作成
        signal = {
            'instrument': instrument,
            'timestamp': timestamp,
            'direction': direction,
            'direction_name': '買い' if direction == 1 else '売り',
            'entry_price': current_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'position_size': position_size,
            'atr': atr,
            'confidence': prediction['confidence'],
            'expected_return': prediction['expected_return'],
            'risk_reward_ratio': abs((take_profit - current_price) / (current_price - stop_loss)) if direction == 1 else abs((current_price - take_profit) / (stop_loss - current_price)),
            'phase1_confidence': prediction['phase1']['confidence'],
            'phase2_return': prediction['phase2']['expected_return']
        }

        # 7. データベース保存
        signal_id = self.db.save_signal(
            timestamp=timestamp,
            instrument=instrument,
            signal_type='BUY' if direction == 1 else 'SELL',
            direction=direction,
            strength=prediction['confidence'],
            price=current_price,
            confidence=prediction['confidence']
        )

        signal['signal_id'] = signal_id

        # ログ
        logger.info(f"  ✅ シグナル生成成功！")
        logger.info(f"     方向: {signal['direction_name']}")
        logger.info(f"     エントリー: {signal['entry_price']:.5f}")
        logger.info(f"     ストップロス: {signal['stop_loss']:.5f}")
        logger.info(f"     テイクプロフィット: {signal['take_profit']:.5f}")
        logger.info(f"     ポジションサイズ: {signal['position_size']:,}単位")
        logger.info(f"     リスク・リワード比: 1:{signal['risk_reward_ratio']:.2f}")
        logger.info(f"     信頼度: {signal['confidence']:.2f}")
        logger.info(f"     期待リターン: {signal['expected_return']:.2f}%")

        return signal

    def _additional_filters(
        self,
        instrument: str,
        prediction: Dict
    ) -> bool:
        """
        追加フィルタリング

        Returns:
        --------
        bool
            True = 合格、False = 不合格
        """
        # 1. ボラティリティチェック（ATR適正範囲）
        atr = self._get_atr(instrument)

        if atr is None:
            return False

        atr_ratio = (atr / prediction['current_price']) * 100

        # ATRが価格の0.3%〜2.0%の範囲にあることを確認
        if atr_ratio < 0.3 or atr_ratio > 2.0:
            logger.info(f"     ATR範囲外: {atr_ratio:.2f}% (0.3%-2.0%)")
            return False

        # 2. 重要ニュースチェック（簡易版）
        # 実際の運用ではニュースカレンダーAPIと連携
        # ここではプレースホルダー
        if self._is_major_news_event(instrument):
            logger.info("     重要ニュースイベント検出")
            return False

        # 3. 市場オープン時間チェック
        if not self._is_market_open(instrument):
            logger.info("     市場クローズ")
            return False

        return True

    def _get_atr(
        self,
        instrument: str,
        period: int = 14
    ) -> Optional[float]:
        """
        ATR（Average True Range）取得

        Returns:
        --------
        float or None
            ATR値
        """
        try:
            # リアルタイムデータから計算
            price_data = self.pipeline.fetch_realtime_data(instrument)

            if 'M1' in price_data and not price_data['M1'].empty:
                df = price_data['M1']

                if len(df) < period + 1:
                    return None

                # TR計算
                high_low = df['high'] - df['low']
                high_close = (df['high'] - df['close'].shift()).abs()
                low_close = (df['low'] - df['close'].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

                # ATR
                atr = tr.rolling(period).mean().iloc[-1]

                return atr

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")

        return None

    def _calculate_position_size(
        self,
        instrument: str,
        entry_price: float,
        stop_loss: float,
        confidence: float,
        expected_return: float
    ) -> int:
        """
        ポジションサイジング

        Kelly基準 + 信頼度調整

        Returns:
        --------
        int
            ポジション単位数
        """
        # リスク額（1単位あたり）
        risk_per_unit = abs(entry_price - stop_loss)

        if risk_per_unit == 0:
            return 0

        # Kelly基準ポジションサイズ
        # f = (p * (b + 1) - 1) / b
        # p = 勝率（信頼度で近似）
        # b = リスク・リワード比率

        win_prob = confidence
        risk_reward = abs(expected_return) / (risk_per_unit / entry_price * 100)

        if risk_reward <= 0:
            return 0

        kelly_f = (win_prob * (risk_reward + 1) - 1) / risk_reward

        # Fractional Kelly（25%）
        kelly_f = kelly_f * self.kelly_fraction

        # 負の値の場合は0
        if kelly_f <= 0:
            return 0

        # 口座残高に対する割合
        position_value = self.account_balance * kelly_f

        # 最大ポジション制約（口座の15%）
        max_position_value = self.account_balance * self.max_position_size_pct
        position_value = min(position_value, max_position_value)

        # 単位数計算
        position_size = int(position_value / entry_price)

        # 最小ポジションサイズ
        position_size = max(position_size, 1000)  # 最低1000単位

        return position_size

    def _is_major_news_event(self, instrument: str) -> bool:
        """
        重要ニュースイベントチェック

        実際の運用ではニュースカレンダーAPIと連携
        """
        # プレースホルダー（常にFalse）
        return False

    def _is_market_open(self, instrument: str) -> bool:
        """
        市場オープンチェック

        FX市場は平日24時間なので、土日以外はTrue
        """
        now = datetime.now()

        # 土日チェック
        if now.weekday() >= 5:  # 5=土曜, 6=日曜
            return False

        return True

    def generate_all_signals(
        self,
        instruments: List[str] = None
    ) -> List[Dict]:
        """
        全通貨ペアのシグナル生成

        Returns:
        --------
        List[Dict]
            生成されたシグナルリスト
        """
        if instruments is None:
            instruments = self.instruments

        signals = []

        for instrument in instruments:
            try:
                signal = self.generate_signal(instrument)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"シグナル生成エラー ({instrument}): {e}")

        logger.info(f"\n生成されたシグナル: {len(signals)}個")

        return signals


if __name__ == "__main__":
    # テスト
    logger.info("シグナル生成システム テスト\n")

    generator = SignalGenerator(
        instruments=['USD_JPY'],
        account_balance=100000.0
    )

    # シグナル生成
    signal = generator.generate_signal('USD_JPY')

    if signal:
        logger.info("\nシグナル詳細:")
        for key, value in signal.items():
            logger.info(f"  {key}: {value}")
    else:
        logger.info("\nシグナルなし（見送り）")
