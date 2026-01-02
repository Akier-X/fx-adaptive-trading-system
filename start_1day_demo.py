"""
1日デモ取引テスト - 仮想1万円

テスト条件:
    - 期間: 24時間
    - チェック間隔: 60分（1時間ごと）
    - 初期資金: 1万円（仮想）
    - 通貨ペア: USD/JPY
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
logger.add(
    f"logs/1day_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG"
)

def main():
    logger.info("=" * 80)
    logger.info("🎮 1日デモ取引テスト開始（仮想1万円）")
    logger.info("=" * 80)
    logger.info(f"開始時刻: {datetime.now()}")
    logger.info("初期資金: ¥10,000（仮想）")
    logger.info("通貨ペア: USD/JPY")
    logger.info("チェック間隔: 60分")
    logger.info("テスト期間: 24時間")
    logger.info("=" * 80)
    logger.info("")
    logger.info("⚠️  これはデモ取引です - 実際のお金は使用しません")
    logger.info("")

    try:
        # ボット起動
        bot = PaperTradingBot(pair='USD/JPY', initial_capital=10000)

        # 24時間連続運用（1時間ごとチェック）
        logger.info("⏰ デモ取引開始...")
        bot.run_continuous(check_interval_minutes=60, duration_days=1)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 1日デモテスト完了！")
        logger.info("=" * 80)

        # 結果サマリー表示
        bot.print_summary()
        bot.save_results()

        logger.info("\n📊 次のステップ:")
        logger.info("  1. 結果確認: python view_results.py")
        logger.info("  2. 問題なければ本番取引: python start_1day_live.py")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ ユーザーによって中断されました")
        if 'bot' in locals():
            bot.print_summary()
            bot.save_results()
    except Exception as e:
        logger.error(f"\n❌ エラー発生: {str(e)}")
        logger.exception(e)
        if 'bot' in locals():
            bot.save_results()
        sys.exit(1)

if __name__ == '__main__':
    main()
