"""
適応学習ボットで1日デモテスト

固定モデルとの比較用
"""

import sys
import time
from datetime import datetime, timedelta
from adaptive_learning_bot import AdaptiveLearningBot
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)
logger.add(
    f"logs/adaptive_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    level="DEBUG"
)


def run_adaptive_demo_test():
    """適応学習ボットで24時間デモテスト"""
    print("\n" + "=" * 80)
    print("適応学習ボット - 1日デモテスト")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("初期資金: 10,000円（仮想）")
    print("通貨ペア: USD/JPY")
    print("チェック間隔: 60分")
    print("テスト期間: 24時間")
    print("")
    print("特徴:")
    print("  - オンライン学習: 50取引ごとに自動更新")
    print("  - ハイブリッド予測: 固定70% + 適応30%")
    print("  - 適応的パラメータ: ボラティリティベース調整")
    print("=" * 80)
    print("")

    try:
        # ボット起動
        bot = AdaptiveLearningBot(pair='USD/JPY', initial_capital=10000)

        # 24時間連続運用
        end_time = datetime.now() + timedelta(days=1)
        cycle = 0
        trades_executed = 0

        print("\n連続運用モード開始...")
        print(f"終了予定: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("")

        while datetime.now() < end_time:
            cycle += 1
            remaining = end_time - datetime.now()

            print("\n" + "=" * 80)
            print(f"サイクル #{cycle}")
            print(f"残り時間: {remaining}")
            print("=" * 80)

            # データ取得
            print("\n1. 最新データ取得中...")
            hist_data = bot.get_historical_data()

            if hist_data is None or len(hist_data) < 50:
                print("  データ取得失敗、次のサイクルへ")
                time.sleep(3600)  # 1時間待機
                continue

            current_price = hist_data['close'].iloc[-1]
            print(f"  現在価格: USD/JPY = {current_price:.2f}円")

            # パラメータ自動調整
            print("\n2. 適応的パラメータ調整...")
            bot.check_and_adapt_parameters(hist_data)

            # 特徴量生成
            print("\n3. 特徴量生成中...")
            features_df = bot.generate_features(hist_data)
            print(f"  特徴量: {len(features_df.columns)}個")

            # ハイブリッド予測
            print("\n4. ハイブリッド予測実行...")
            signal = bot.predict_signal(features_df)

            print(f"\n予測結果:")
            print(f"  方向: {'上昇' if signal['direction'] == 1 else '下降'}")
            print(f"  信頼度: {signal['confidence']:.4f} ({signal['confidence']*100:.2f}%)")
            print(f"  期待リターン: {signal['expected_return']:.4f}%")

            if signal['online_pred'] is not None:
                print(f"\n  固定モデル: 信頼度 {signal['fixed_pred']['confidence']:.4f}")
                print(f"  適応モデル: 信頼度 {signal['online_pred']['confidence']:.4f}")
                print(f"  ハイブリッド統合済み")
            else:
                print(f"\n  固定モデルのみ（適応モデル未訓練）")

            # 取引判定
            will_trade = (
                signal['confidence'] >= bot.phase1_confidence_threshold and
                abs(signal['expected_return']) >= bot.phase2_min_return
            )

            print(f"\n5. 取引判定:")
            print(f"  信頼度: {signal['confidence']:.4f} >= {bot.phase1_confidence_threshold} ? {'YES' if signal['confidence'] >= bot.phase1_confidence_threshold else 'NO'}")
            print(f"  期待リターン: {abs(signal['expected_return']):.4f} >= {bot.phase2_min_return} ? {'YES' if abs(signal['expected_return']) >= bot.phase2_min_return else 'NO'}")

            if will_trade:
                print(f"\n  >>> 取引実行（シミュレーション） <<<")
                trades_executed += 1

                # シミュレーション取引（実際には執行しない）
                direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'
                print(f"  方向: {direction_str}")
                print(f"  エントリー: {current_price:.2f}円")

                # 次のサイクルで結果をシミュレート
                # （実際の実装ではポジション管理が必要）

                # オンライン学習用にデータ記録
                # 注: 実際のラベル（上昇/下降）は次の価格データが必要
                # ここでは予測方向を仮ラベルとして使用（デモ用）
                latest_features = features_df.iloc[-1:].values.reshape(1, -1)
                bot.update_online_models(latest_features, signal['direction'])

            else:
                print(f"\n  >>> 取引見送り <<<")

                reasons = []
                if signal['confidence'] < bot.phase1_confidence_threshold:
                    diff = (bot.phase1_confidence_threshold - signal['confidence']) * 100
                    reasons.append(f"信頼度不足（あと{diff:.2f}%必要）")
                if abs(signal['expected_return']) < bot.phase2_min_return:
                    diff = bot.phase2_min_return - abs(signal['expected_return'])
                    reasons.append(f"期待リターン不足（あと{diff:.4f}%必要）")

                print(f"  理由: {', '.join(reasons)}")

            # 統計表示
            print(f"\n累計統計:")
            print(f"  サイクル数: {cycle}")
            print(f"  取引実行: {trades_executed}回")
            print(f"  オンライン学習: {'訓練済み' if bot.online_model_trained else '未訓練'}（{len(bot.update_buffer)}/50サンプル）")

            # 次のサイクルまで待機
            if datetime.now() < end_time:
                print(f"\n次回実行: {(datetime.now() + timedelta(hours=1)).strftime('%H:%M:%S')}")
                print(f"60分間待機中...")
                time.sleep(3600)  # 1時間待機

        # テスト完了
        print("\n" + "=" * 80)
        print("1日適応学習デモテスト完了！")
        print("=" * 80)
        print(f"総サイクル数: {cycle}")
        print(f"総取引数: {trades_executed}")
        print(f"オンライン学習モデル: {'訓練完了' if bot.online_model_trained else '未訓練'}")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\nユーザーによって中断されました")
        print(f"実行サイクル数: {cycle if 'cycle' in locals() else 0}")
        print(f"取引数: {trades_executed if 'trades_executed' in locals() else 0}")
    except Exception as e:
        print(f"\nエラー発生: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_adaptive_demo_test()
