"""
デモ取引結果レビュー & 本番移行判定

デモテストの結果を分析し、本番取引に移行すべきか判定します。
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def find_latest_demo_results():
    """最新のデモ結果を探す"""
    output_dir = Path('outputs/paper_trading')

    if not output_dir.exists():
        return None

    summary_files = list(output_dir.glob('summary_*.json'))
    if not summary_files:
        return None

    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    timestamp = latest_summary.stem.replace('summary_', '')

    return {
        'summary': latest_summary,
        'trades': output_dir / f'trades_{timestamp}.csv',
        'equity': output_dir / f'equity_{timestamp}.csv',
        'timestamp': timestamp
    }

def analyze_results(files):
    """結果を分析して本番移行判定"""
    print("\n" + "=" * 80)
    print("📊 デモ取引結果レビュー")
    print("=" * 80)

    # サマリー読み込み
    with open(files['summary'], 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # 基本情報
    print(f"\n⏰ テスト期間: {summary.get('start_time', 'N/A')} 〜 {summary.get('end_time', 'N/A')}")
    print(f"通貨ペア: {summary.get('pair', 'USD/JPY')}")

    # 資金状況
    initial = summary.get('initial_capital', 10000)
    final = summary.get('final_capital', 0)
    pnl = summary.get('total_pnl', 0)
    pnl_pct = summary.get('return_pct', 0)

    print(f"\n💰 資金状況:")
    print(f"  初期資金: ¥{initial:,.0f}")
    print(f"  最終資金: ¥{final:,.2f}")

    if pnl >= 0:
        print(f"  総損益: +¥{pnl:,.2f} (+{pnl_pct:.2f}%) ✅")
    else:
        print(f"  総損益: -¥{abs(pnl):,.2f} ({pnl_pct:.2f}%) ❌")

    # 取引統計
    total_trades = summary.get('total_trades', 0)
    win_rate = summary.get('win_rate', 0)
    profit_factor = summary.get('profit_factor', 0)
    max_dd_pct = summary.get('max_drawdown_pct', 0)

    print(f"\n📈 取引統計:")
    print(f"  総取引数: {total_trades}回")
    print(f"  勝率: {win_rate:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  最大DD: {max_dd_pct:.2f}%")

    # 判定基準
    print("\n" + "=" * 80)
    print("🎯 本番移行判定基準")
    print("=" * 80)

    criteria = {
        '総損益がプラス': pnl > 0,
        '勝率 >= 70%': win_rate >= 70,
        '取引回数 >= 1回': total_trades >= 1,
        'Profit Factor >= 1.5': profit_factor >= 1.5,
        '最大DD < 15%': abs(max_dd_pct) < 15
    }

    passed = 0
    total = len(criteria)

    for criterion, result in criteria.items():
        status = "✅ 合格" if result else "❌ 不合格"
        print(f"  {criterion}: {status}")
        if result:
            passed += 1

    # 総合判定
    print("\n" + "=" * 80)
    print(f"総合スコア: {passed}/{total}")

    if passed == total:
        print("判定: ✅ **本番取引に移行可能** ✅")
        recommendation = "proceed"
    elif passed >= total * 0.8:
        print("判定: ⚠️ **条件付きで移行可能** ⚠️")
        print("      より長期間のテストを推奨")
        recommendation = "caution"
    else:
        print("判定: ❌ **本番移行は推奨しません** ❌")
        print("      パラメータ調整または追加テストが必要")
        recommendation = "stop"

    print("=" * 80)

    # 推奨アクション
    print("\n📋 推奨アクション:")

    if recommendation == "proceed":
        print("  ✅ 本番取引を開始できます")
        print("  ✅ 次のコマンド: python start_1day_live.py")
        print("")
        print("  ⚠️ 注意事項:")
        print("     - .envで OANDA_ENVIRONMENT=live に変更")
        print("     - OANDA本番アカウントのトークンを設定")
        print("     - 初回は1万円で慎重にスタート")
        print("")
    elif recommendation == "caution":
        print("  ⚠️ より長期間のデモテストを推奨")
        print("  ⚠️ 次のコマンド: python start_1week_test.py")
        print("")
        print("  または:")
        print("  ⚠️ 非常に少額（5,000円など）で本番テスト")
        print("")
    else:
        print("  ❌ 本番移行は推奨しません")
        print("  ❌ 以下を検討してください:")
        print("     1. パラメータ調整（Kelly係数を下げる等）")
        print("     2. より長期間のデモテスト")
        print("     3. モデルの再訓練")
        print("")

    # 詳細な取引履歴
    if files['trades'].exists():
        trades_df = pd.read_csv(files['trades'])

        if len(trades_df) > 0:
            print("\n📋 全取引履歴:")
            print("-" * 80)

            for idx, trade in trades_df.iterrows():
                direction = "🟢 LONG" if trade.get('direction', '') == 'LONG' else "🔴 SHORT"
                pnl_trade = trade.get('pnl', 0)
                pnl_sign = "+" if pnl_trade >= 0 else ""
                result_icon = "✅" if pnl_trade >= 0 else "❌"

                print(f"{result_icon} {trade.get('entry_time', 'N/A')} | {direction} | "
                      f"エントリー: ¥{trade.get('entry_price', 0):.2f} → "
                      f"決済: ¥{trade.get('exit_price', 0):.2f} | "
                      f"損益: {pnl_sign}¥{pnl_trade:,.2f} ({pnl_sign}{trade.get('pnl_pct', 0):.2f}%)")

    print("\n" + "=" * 80 + "\n")

    return recommendation

def main():
    files = find_latest_demo_results()

    if not files:
        print("\n❌ デモテストの結果が見つかりません。")
        print("\nまずデモテストを実行してください:")
        print("  python start_1day_demo.py")
        return

    recommendation = analyze_results(files)

    # 次のステップを明確に提示
    print("=" * 80)
    print("🚀 次のステップ")
    print("=" * 80)

    if recommendation == "proceed":
        print("\n1. .envファイルを編集:")
        print("   OANDA_ENVIRONMENT=live")
        print("")
        print("2. OANDA本番アカウントのトークンを設定:")
        print("   OANDA_ACCESS_TOKEN=your_live_token_here")
        print("")
        print("3. 本番取引開始:")
        print("   python start_1day_live.py")
        print("")
    elif recommendation == "caution":
        print("\n推奨: より長期間のテスト")
        print("  python start_1week_test.py")
        print("")
    else:
        print("\n推奨: パラメータ調整またはモデル再訓練")
        print("")

    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
