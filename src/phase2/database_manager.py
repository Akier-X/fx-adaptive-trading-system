"""
Phase 2 データベース管理

リアルタイムデータ・予測・トレード履歴の保存
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import json
from loguru import logger


class DatabaseManager:
    """
    Phase 2 データベースマネージャー

    テーブル構成:
    1. price_data - 価格データ（1分足）
    2. predictions - 予測結果
    3. trades - 取引履歴
    4. positions - ポジション管理
    5. performance - パフォーマンス指標
    6. signals - シグナル履歴
    """

    def __init__(self, db_path: Path = Path("data/phase2/trading.db")):
        """
        Parameters:
        -----------
        db_path : Path
            データベースファイルパス
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # データベース接続
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False
        )

        # テーブル作成
        self._create_tables()

        logger.info(f"データベース初期化完了: {self.db_path}")

    def _create_tables(self):
        """全テーブル作成"""

        # 1. 価格データテーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS price_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                granularity TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                spread REAL,
                UNIQUE(timestamp, instrument, granularity)
            )
        """)

        # 2. 予測テーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prediction_direction INTEGER,
                prediction_probability REAL,
                confidence REAL,
                target_price REAL,
                current_price REAL,
                actual_direction INTEGER,
                actual_return REAL,
                correct BOOLEAN,
                features_json TEXT
            )
        """)

        # 3. トレードテーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                units INTEGER NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                pnl REAL,
                pnl_pct REAL,
                status TEXT,
                signal_id INTEGER,
                FOREIGN KEY(signal_id) REFERENCES signals(id)
            )
        """)

        # 4. ポジションテーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                units INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT NOT NULL
            )
        """)

        # 5. パフォーマンステーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                balance REAL NOT NULL,
                equity REAL NOT NULL,
                unrealized_pnl REAL NOT NULL,
                daily_return REAL,
                cumulative_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                profit_factor REAL,
                total_trades INTEGER
            )
        """)

        # 6. シグナルテーブル
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                instrument TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                direction INTEGER NOT NULL,
                strength REAL NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                regime TEXT,
                models_agreement REAL,
                executed BOOLEAN DEFAULT 0
            )
        """)

        self.conn.commit()
        logger.info("全テーブル作成完了")

    def save_price_data(
        self,
        timestamp: datetime,
        instrument: str,
        granularity: str,
        ohlcv: Dict[str, float]
    ):
        """
        価格データ保存

        Parameters:
        -----------
        timestamp : datetime
            タイムスタンプ
        instrument : str
            通貨ペア
        granularity : str
            時間足
        ohlcv : Dict
            {'open', 'high', 'low', 'close', 'volume', 'spread'}
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO price_data
                (timestamp, instrument, granularity, open, high, low, close, volume, spread)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                instrument,
                granularity,
                ohlcv['open'],
                ohlcv['high'],
                ohlcv['low'],
                ohlcv['close'],
                ohlcv.get('volume', 0),
                ohlcv.get('spread', 0)
            ))
            self.conn.commit()

        except Exception as e:
            logger.error(f"価格データ保存エラー: {e}")

    def save_prediction(
        self,
        timestamp: datetime,
        instrument: str,
        model_name: str,
        prediction: Dict
    ) -> int:
        """
        予測結果保存

        Parameters:
        -----------
        timestamp : datetime
            予測時刻
        instrument : str
            通貨ペア
        model_name : str
            モデル名
        prediction : Dict
            予測結果

        Returns:
        --------
        int
            予測ID
        """
        try:
            cursor = self.conn.execute("""
                INSERT INTO predictions
                (timestamp, instrument, model_name, prediction_direction,
                 prediction_probability, confidence, target_price, current_price,
                 features_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp,
                instrument,
                model_name,
                prediction.get('direction', 0),
                prediction.get('probability', 0.5),
                prediction.get('confidence', 0.5),
                prediction.get('target_price', 0),
                prediction.get('current_price', 0),
                json.dumps(prediction.get('features', {}))
            ))
            self.conn.commit()

            return cursor.lastrowid

        except Exception as e:
            logger.error(f"予測保存エラー: {e}")
            return -1

    def update_prediction_result(
        self,
        prediction_id: int,
        actual_direction: int,
        actual_return: float
    ):
        """
        予測結果の答え合わせ

        Parameters:
        -----------
        prediction_id : int
            予測ID
        actual_direction : int
            実際の方向（1=上昇, -1=下降）
        actual_return : float
            実際のリターン（%）
        """
        try:
            # 予測データ取得
            cursor = self.conn.execute("""
                SELECT prediction_direction FROM predictions WHERE id = ?
            """, (prediction_id,))

            result = cursor.fetchone()
            if result:
                predicted_direction = result[0]
                correct = (predicted_direction == actual_direction)

                # 更新
                self.conn.execute("""
                    UPDATE predictions
                    SET actual_direction = ?,
                        actual_return = ?,
                        correct = ?
                    WHERE id = ?
                """, (actual_direction, actual_return, correct, prediction_id))

                self.conn.commit()

        except Exception as e:
            logger.error(f"予測結果更新エラー: {e}")

    def save_trade(
        self,
        timestamp: datetime,
        instrument: str,
        direction: str,
        entry_price: float,
        units: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        signal_id: Optional[int] = None
    ) -> int:
        """
        トレード保存

        Parameters:
        -----------
        timestamp : datetime
            エントリー時刻
        instrument : str
            通貨ペア
        direction : str
            'LONG' or 'SHORT'
        entry_price : float
            エントリー価格
        units : int
            ユニット数
        stop_loss : float, optional
            ストップロス
        take_profit : float, optional
            テイクプロフィット
        signal_id : int, optional
            シグナルID

        Returns:
        --------
        int
            トレードID
        """
        try:
            cursor = self.conn.execute("""
                INSERT INTO trades
                (timestamp, instrument, direction, entry_price, units,
                 stop_loss, take_profit, entry_time, status, signal_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)
            """, (
                timestamp,
                instrument,
                direction,
                entry_price,
                units,
                stop_loss,
                take_profit,
                timestamp,
                signal_id
            ))
            self.conn.commit()

            return cursor.lastrowid

        except Exception as e:
            logger.error(f"トレード保存エラー: {e}")
            return -1

    def close_trade(
        self,
        trade_id: int,
        exit_price: float,
        exit_time: datetime
    ):
        """
        トレードクローズ

        Parameters:
        -----------
        trade_id : int
            トレードID
        exit_price : float
            決済価格
        exit_time : datetime
            決済時刻
        """
        try:
            # トレード情報取得
            cursor = self.conn.execute("""
                SELECT direction, entry_price, units FROM trades WHERE id = ?
            """, (trade_id,))

            result = cursor.fetchone()
            if result:
                direction, entry_price, units = result

                # PnL計算
                if direction == 'LONG':
                    pnl = (exit_price - entry_price) * units
                    pnl_pct = ((exit_price / entry_price) - 1) * 100
                else:  # SHORT
                    pnl = (entry_price - exit_price) * units
                    pnl_pct = ((entry_price / exit_price) - 1) * 100

                # 更新
                self.conn.execute("""
                    UPDATE trades
                    SET exit_price = ?,
                        exit_time = ?,
                        pnl = ?,
                        pnl_pct = ?,
                        status = 'CLOSED'
                    WHERE id = ?
                """, (exit_price, exit_time, pnl, pnl_pct, trade_id))

                self.conn.commit()

                logger.info(f"トレードクローズ: ID={trade_id}, PnL={pnl:.2f} ({pnl_pct:.2f}%)")

        except Exception as e:
            logger.error(f"トレードクローズエラー: {e}")

    def save_signal(
        self,
        timestamp: datetime,
        instrument: str,
        signal_type: str,
        direction: int,
        strength: float,
        price: float,
        confidence: float,
        regime: str = None,
        models_agreement: float = 0.0
    ) -> int:
        """
        シグナル保存

        Returns:
        --------
        int
            シグナルID
        """
        try:
            cursor = self.conn.execute("""
                INSERT INTO signals
                (timestamp, instrument, signal_type, direction, strength,
                 price, confidence, regime, models_agreement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timestamp, instrument, signal_type, direction, strength,
                price, confidence, regime, models_agreement
            ))
            self.conn.commit()

            return cursor.lastrowid

        except Exception as e:
            logger.error(f"シグナル保存エラー: {e}")
            return -1

    def get_recent_predictions(
        self,
        instrument: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        最近の予測取得

        Returns:
        --------
        pd.DataFrame
            予測履歴
        """
        query = """
            SELECT * FROM predictions
            WHERE instrument = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """

        return pd.read_sql_query(
            query,
            self.conn,
            params=(instrument, limit)
        )

    def get_trade_history(
        self,
        instrument: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        トレード履歴取得

        Returns:
        --------
        pd.DataFrame
            トレード履歴
        """
        if instrument:
            query = """
                SELECT * FROM trades
                WHERE instrument = ?
                ORDER BY entry_time DESC
                LIMIT ?
            """
            params = (instrument, limit)
        else:
            query = """
                SELECT * FROM trades
                ORDER BY entry_time DESC
                LIMIT ?
            """
            params = (limit,)

        return pd.read_sql_query(query, self.conn, params=params)

    def get_performance_stats(self) -> Dict:
        """
        パフォーマンス統計取得

        Returns:
        --------
        Dict
            統計情報
        """
        try:
            # トレード統計
            trades_df = self.get_trade_history(limit=1000)

            if trades_df.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0
                }

            closed_trades = trades_df[trades_df['status'] == 'CLOSED']

            if closed_trades.empty:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0
                }

            wins = len(closed_trades[closed_trades['pnl'] > 0])
            losses = len(closed_trades[closed_trades['pnl'] <= 0])
            total = len(closed_trades)

            return {
                'total_trades': total,
                'wins': wins,
                'losses': losses,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'avg_pnl': closed_trades['pnl'].mean(),
                'total_pnl': closed_trades['pnl'].sum(),
                'avg_pnl_pct': closed_trades['pnl_pct'].mean(),
                'max_win': closed_trades['pnl'].max(),
                'max_loss': closed_trades['pnl'].min()
            }

        except Exception as e:
            logger.error(f"パフォーマンス統計エラー: {e}")
            return {}

    def get_prediction_accuracy(
        self,
        instrument: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> Dict:
        """
        予測精度取得

        Returns:
        --------
        Dict
            精度統計
        """
        try:
            query = "SELECT * FROM predictions WHERE actual_direction IS NOT NULL"
            params = []

            if instrument:
                query += " AND instrument = ?"
                params.append(instrument)

            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)

            df = pd.read_sql_query(query, self.conn, params=params)

            if df.empty:
                return {'accuracy': 0, 'total': 0}

            correct = df['correct'].sum()
            total = len(df)

            return {
                'accuracy': (correct / total * 100) if total > 0 else 0,
                'total': total,
                'correct': correct,
                'incorrect': total - correct
            }

        except Exception as e:
            logger.error(f"予測精度取得エラー: {e}")
            return {}

    def close(self):
        """データベース接続クローズ"""
        if self.conn:
            self.conn.close()
            logger.info("データベース接続クローズ")


if __name__ == "__main__":
    # テスト
    db = DatabaseManager()

    # 価格データ保存テスト
    db.save_price_data(
        timestamp=datetime.now(),
        instrument='USD_JPY',
        granularity='M1',
        ohlcv={
            'open': 150.00,
            'high': 150.10,
            'low': 149.90,
            'close': 150.05,
            'volume': 1000,
            'spread': 0.002
        }
    )

    # 予測保存テスト
    pred_id = db.save_prediction(
        timestamp=datetime.now(),
        instrument='USD_JPY',
        model_name='phase1_8',
        prediction={
            'direction': 1,
            'probability': 0.92,
            'confidence': 0.85,
            'current_price': 150.00
        }
    )

    logger.info(f"予測ID: {pred_id}")

    # パフォーマンス統計
    stats = db.get_performance_stats()
    logger.info(f"統計: {stats}")
