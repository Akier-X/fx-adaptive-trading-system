"""
予測結果のリアルタイム表示（シンプル版）
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime
from paper_trading_bot import PaperTradingBot

def show_current_prediction():
    """現在の予測を表示"""
    print("\n" + "=" * 80)
    print("現在の市場予測")
    print("=" * 80)
    print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    try:
        # ボット初期化
        print("モデル読み込み中...")
        bot = PaperTradingBot(pair='USD/JPY', initial_capital=10000)
        print("モデル読み込み完了")
        print("")

        # 現在価格取得
        print("最新データ取得中...")
        hist_data_temp = bot.get_historical_data()
        current_price = hist_data_temp['close'].iloc[-1]
        print(f"  現在価格: USD/JPY = {current_price:.2f}円")
        print("")

        # 履歴データ取得
        print("過去データ取得中...")
        hist_data = bot.get_historical_data()
        print(f"  過去データ: {len(hist_data)}日分")
        print("")

        # 特徴量生成
        print("特徴量生成中...")
        features_df = bot.generate_features(hist_data)
        print(f"  特徴量数: {len(features_df.columns)}個")
        print("")

        # 予測実行
        print("予測実行中...")
        signal = bot.predict_signal(features_df)
        print("")

        # 結果表示
        print("=" * 80)
        print("予測結果")
        print("=" * 80)

        print(f"\n[Phase 1.8 - 方向予測]")
        print(f"  信頼度: {signal['confidence']:.4f} ({signal['confidence']*100:.2f}%)")
        print(f"  閾値: {bot.phase1_confidence_threshold} ({bot.phase1_confidence_threshold*100:.0f}%)")
        print(f"  方向: {'上昇' if signal['direction'] == 1 else '下降'}")

        if signal['confidence'] >= bot.phase1_confidence_threshold:
            print(f"  判定: OK - 信頼度条件を満たしています")
        else:
            print(f"  判定: NG - 信頼度不足 (差: {(bot.phase1_confidence_threshold - signal['confidence'])*100:.2f}%)")

        print(f"\n[Phase 2 - 収益予測]")
        print(f"  期待リターン: {signal['expected_return']:.4f}%")
        print(f"  閾値: {bot.phase2_min_return}%")

        if abs(signal['expected_return']) >= bot.phase2_min_return:
            print(f"  判定: OK - 期待リターン条件を満たしています")
        else:
            print(f"  判定: NG - 期待リターン不足 (差: {(bot.phase2_min_return - abs(signal['expected_return'])):.4f}%)")

        print(f"\n[ハイブリッド判定]")

        # 取引判定
        will_trade = (
            signal['confidence'] >= bot.phase1_confidence_threshold and
            abs(signal['expected_return']) >= bot.phase2_min_return
        )

        if will_trade:
            print(f"  >>> 取引実行 <<<")

            # ポジションサイズ計算
            position_size = bot.calculate_position_size(signal)
            position_value = bot.current_capital * position_size

            print(f"\n[取引詳細]")
            print(f"  ポジションサイズ: {position_size*100:.2f}%")
            print(f"  取引金額: {position_value:,.0f}円")
            print(f"  方向: {'LONG（買い）' if signal['direction'] == 1 else 'SHORT（売り）'}")
            print(f"  エントリー価格: {current_price:.2f}円")

        else:
            print(f"  >>> 取引見送り <<<")

            # 理由を詳細に
            print(f"\n  見送り理由:")
            if signal['confidence'] < bot.phase1_confidence_threshold:
                diff = (bot.phase1_confidence_threshold - signal['confidence']) * 100
                print(f"    [NG] Phase 1信頼度不足")
                print(f"         現在: {signal['confidence']*100:.2f}%")
                print(f"         必要: {bot.phase1_confidence_threshold*100:.0f}%")
                print(f"         差分: あと{diff:.2f}%必要")
            else:
                print(f"    [OK] Phase 1信頼度: 条件クリア")

            if abs(signal['expected_return']) < bot.phase2_min_return:
                diff = bot.phase2_min_return - abs(signal['expected_return'])
                print(f"    [NG] Phase 2期待リターン不足")
                print(f"         現在: {abs(signal['expected_return']):.4f}%")
                print(f"         必要: {bot.phase2_min_return}%")
                print(f"         差分: あと{diff:.4f}%必要")
            else:
                print(f"    [OK] Phase 2期待リターン: 条件クリア")

        print("\n" + "=" * 80)

        # 過去24時間の統計
        print("\n過去24時間の価格変動")
        print("=" * 80)

        recent_data = hist_data.tail(24)
        if len(recent_data) >= 2:
            price_change = ((recent_data['close'].iloc[-1] / recent_data['close'].iloc[0]) - 1) * 100
            volatility = recent_data['close'].pct_change().std() * 100

            print(f"  24時間変動: {price_change:+.2f}%")
            print(f"  ボラティリティ: {volatility:.2f}%")
            print(f"  最高値: {recent_data['high'].max():.2f}円")
            print(f"  最安値: {recent_data['low'].min():.2f}円")
            print(f"  現在値: {recent_data['close'].iloc[-1]:.2f}円")

        print("\n" + "=" * 80)
        print("")
        print("この予測は現在のモデル（訓練済み）を使用しています。")
        print("リアルタイム学習・適応は未実装です。")
        print("")

    except Exception as e:
        print(f"\nエラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    show_current_prediction()
