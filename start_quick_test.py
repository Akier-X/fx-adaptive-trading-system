"""
クイックテスト - 1時間版
1週間テストの前に動作確認用

テスト条件:
    - 期間: 1時間（10分ごとチェック）
    - 初期資金: 1万円
    - すぐに結果確認可能
"""

import sys
from datetime import datetime
from paper_trading_bot import PaperTradingBot
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)

def main():
    logger.info("=" * 80)
    logger.info("⚡ クイックテスト開始（1時間、10分ごとチェック）")
    logger.info("=" * 80)

    bot = PaperTradingBot(pair='USD/JPY', initial_capital=10000)

    # 1時間テスト（10分ごと = 6回チェック）
    logger.info("⏰ 1時間テスト開始...")
    bot.run_continuous(check_interval_minutes=10, duration_days=1/24)  # 1/24日 = 1時間

    logger.info("\n✅ クイックテスト完了！")
    bot.print_summary()
    bot.save_results()

if __name__ == '__main__':
    main()
