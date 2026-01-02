"""
1日本番取引 - 実資金1万円

⚠️ 警告: これは実際のお金を使用します！

実行前チェックリスト:
    ✅ デモテストで良好な結果を確認済み
    ✅ OANDA本番アカウント準備完了
    ✅ .envで OANDA_ENVIRONMENT=live に設定
    ✅ 損失を許容できる範囲の資金のみ使用

テスト条件:
    - 期間: 24時間
    - チェック間隔: 60分（1時間ごと）
    - 初期資金: 1万円（実資金）
    - 通貨ペア: USD/JPY
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
    f"logs/1day_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG"
)

def confirm_live_trading():
    """本番取引の確認"""
    print("\n" + "=" * 80)
    print("⚠️  警告: 本番取引モード")
    print("=" * 80)
    print("")
    print("これは実際のお金を使用する取引です。")
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
    print("  ✅ デモテストで良好な結果を確認済み")
    print("  ✅ OANDA本番アカウント準備完了")
    print("  ✅ 損失を許容できる範囲の資金のみ使用")
    print("")

    response = input("本番取引を開始しますか？ (yes/no): ")

    if response.lower() != 'yes':
        print("\n取引をキャンセルしました。")
        return False

    # 再確認
    print("")
    response2 = input("本当に実行しますか？この操作は取り消せません。 (YES/no): ")

    return response2 == 'YES'

def main():
    # 本番取引確認
    if not confirm_live_trading():
        logger.info("本番取引をキャンセルしました。")
        sys.exit(0)

    logger.info("=" * 80)
    logger.info("💰 1日本番取引開始（実資金1万円）")
    logger.info("=" * 80)
    logger.info(f"開始時刻: {datetime.now()}")
    logger.info("初期資金: ¥10,000（実資金）")
    logger.info("通貨ペア: USD/JPY")
    logger.info("チェック間隔: 60分")
    logger.info("テスト期間: 24時間")
    logger.info("=" * 80)
    logger.info("")
    logger.warning("⚠️  本番取引モード - 実際のお金を使用します")
    logger.info("")

    try:
        # ボット起動
        bot = LiveTradingBot(pair='USD/JPY', initial_capital=10000)

        # 24時間連続運用（1時間ごとチェック）
        logger.info("⏰ 本番取引開始...")
        bot.run_continuous(check_interval_minutes=60, duration_days=1)

        logger.info("\n" + "=" * 80)
        logger.info("✅ 1日本番取引完了！")
        logger.info("=" * 80)

        logger.info("\n📊 次のステップ:")
        logger.info("  1. OANDA口座で実際の取引結果を確認")
        logger.info("  2. 結果が良好なら: python start_continuous_50k.py")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ ユーザーによって中断されました")
        logger.warning("⚠️ OANDA口座で開いているポジションを確認してください")
    except Exception as e:
        logger.error(f"\n❌ エラー発生: {str(e)}")
        logger.exception(e)
        logger.error("⚠️ OANDA口座で開いているポジションを確認してください")
        sys.exit(1)

if __name__ == '__main__':
    main()
