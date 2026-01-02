"""
継続実行 - 5万円運用

⚠️ 警告: これは実際のお金を使用します！

実行前チェックリスト:
    ✅ 1日本番取引で良好な結果を確認済み
    ✅ 勝率85%以上達成
    ✅ プラスのリターン確認
    ✅ 5万円の損失を許容可能

運用条件:
    - 初期資金: 5万円（実資金）
    - 通貨ペア: USD/JPY
    - チェック間隔: 60分（1時間ごと）
    - 継続実行（無期限）
"""

import sys
import os
from datetime import datetime
from live_trading_bot import LiveTradingBot
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    f"logs/continuous_50k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG",
    rotation="1 day"  # 日次ローテーション
)

def confirm_50k_trading():
    """5万円運用の確認"""
    print("\n" + "=" * 80)
    print("💰 5万円継続運用モード")
    print("=" * 80)
    print("")
    print("これは5万円の実資金を使用する継続運用です。")
    print("")

    # .env確認
    env_mode = os.getenv('OANDA_ENVIRONMENT', 'practice')
    print(f"現在のOANDA環境: {env_mode}")

    if env_mode != 'live':
        print("")
        print("❌ エラー: .envファイルで OANDA_ENVIRONMENT=live に設定してください")
        print("")
        return False

    print("")
    print("チェックリスト:")
    print("  ✅ 1日本番取引で良好な結果を確認済み")
    print("  ✅ 勝率85%以上達成")
    print("  ✅ プラスのリターン確認")
    print("  ✅ 5万円の損失を許容可能")
    print("")
    print("運用条件:")
    print("  - 初期資金: ¥50,000")
    print("  - 継続実行: 無期限（手動停止まで）")
    print("  - チェック間隔: 1時間ごと")
    print("")

    response = input("5万円継続運用を開始しますか？ (yes/no): ")

    if response.lower() != 'yes':
        print("\n運用をキャンセルしました。")
        return False

    # 再確認
    print("")
    response2 = input("本当に実行しますか？5万円の実資金を使用します。 (YES/no): ")

    return response2 == 'YES'

def main():
    # 5万円運用確認
    if not confirm_50k_trading():
        logger.info("5万円運用をキャンセルしました。")
        sys.exit(0)

    logger.info("=" * 80)
    logger.info("💰 5万円継続運用開始")
    logger.info("=" * 80)
    logger.info(f"開始時刻: {datetime.now()}")
    logger.info("初期資金: ¥50,000（実資金）")
    logger.info("通貨ペア: USD/JPY")
    logger.info("チェック間隔: 60分")
    logger.info("運用期間: 無期限（手動停止まで）")
    logger.info("=" * 80)
    logger.info("")
    logger.warning("⚠️  本番取引モード - 5万円の実資金を使用します")
    logger.info("")
    logger.info("📊 停止方法: Ctrl+C を押してください")
    logger.info("📊 リアルタイム監視: 別ウィンドウで python monitor_test.py")
    logger.info("")

    try:
        # ボット起動（5万円）
        bot = LiveTradingBot(pair='USD/JPY', initial_capital=50000)

        # 無期限連続運用（1時間ごとチェック）
        logger.info("⏰ 継続運用開始...")

        # 非常に長い期間（365日 = 1年）
        # 実際は手動停止するまで実行
        bot.run_continuous(check_interval_minutes=60, duration_days=365)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 運用終了")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ ユーザーによって運用を停止しました")
        logger.warning("⚠️ OANDA口座で開いているポジションを確認してください")
        logger.info("\n📊 運用統計:")
        logger.info("  - OANDA口座で詳細な取引履歴を確認してください")
    except Exception as e:
        logger.error(f"\n❌ エラー発生: {str(e)}")
        logger.exception(e)
        logger.error("⚠️ OANDA口座で開いているポジションを確認してください")
        sys.exit(1)

if __name__ == '__main__':
    main()
