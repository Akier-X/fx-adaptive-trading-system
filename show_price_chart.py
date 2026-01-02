"""
USD/JPY価格推移グラフ表示

過去のデータと現在の価格をグラフ化
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

def create_price_chart():
    """USD/JPY価格推移グラフを作成"""

    print("\n" + "=" * 80)
    print("USD/JPY 価格推移グラフ作成中...")
    print("=" * 80)

    try:
        # データ取得（過去6ヶ月）
        print("\nデータ取得中...")
        ticker = yf.Ticker("USDJPY=X")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # 6ヶ月

        data = ticker.history(start=start_date, end=end_date, interval='1d')

        if data.empty:
            print("エラー: データが取得できませんでした")
            return

        # USD/JPYの価格（そのまま使用）
        data['USD_JPY'] = data['Close']

        print(f"データ取得完了: {len(data)}日分")
        print(f"期間: {data.index[0].strftime('%Y-%m-%d')} ~ {data.index[-1].strftime('%Y-%m-%d')}")
        print(f"現在価格: {data['USD_JPY'].iloc[-1]:.2f}円")

        # 移動平均計算
        data['MA_7'] = data['USD_JPY'].rolling(window=7).mean()
        data['MA_25'] = data['USD_JPY'].rolling(window=25).mean()
        data['MA_50'] = data['USD_JPY'].rolling(window=50).mean()

        # ボリンジャーバンド
        data['BB_upper'] = data['MA_25'] + (data['USD_JPY'].rolling(window=25).std() * 2)
        data['BB_lower'] = data['MA_25'] - (data['USD_JPY'].rolling(window=25).std() * 2)

        # グラフ作成
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('USD/JPY 価格推移分析', fontsize=20, fontweight='bold', y=0.995)

        # === グラフ1: 価格推移 + 移動平均 ===
        ax1 = axes[0]
        ax1.plot(data.index, data['USD_JPY'], label='USD/JPY', color='#2E86AB', linewidth=2)
        ax1.plot(data.index, data['MA_7'], label='7日移動平均', color='#F77F00', linewidth=1.5, alpha=0.7)
        ax1.plot(data.index, data['MA_25'], label='25日移動平均', color='#06A77D', linewidth=1.5, alpha=0.7)
        ax1.plot(data.index, data['MA_50'], label='50日移動平均', color='#D62828', linewidth=1.5, alpha=0.7)

        # ボリンジャーバンド
        ax1.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1, color='gray', label='ボリンジャーバンド(±2σ)')

        ax1.set_ylabel('価格 (円)', fontsize=12, fontweight='bold')
        ax1.set_title('価格推移と移動平均線', fontsize=14, fontweight='bold', pad=10)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 最新価格をマーク
        latest_price = data['USD_JPY'].iloc[-1]
        latest_date = data.index[-1]
        ax1.scatter([latest_date], [latest_price], color='red', s=100, zorder=5, marker='o')
        ax1.annotate(f'現在: {latest_price:.2f}円',
                     xy=(latest_date, latest_price),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     fontsize=10, fontweight='bold')

        # === グラフ2: 日次変動率 ===
        ax2 = axes[1]
        daily_returns = data['USD_JPY'].pct_change() * 100
        colors = ['green' if x > 0 else 'red' for x in daily_returns]
        ax2.bar(data.index, daily_returns, color=colors, alpha=0.6, width=1.0)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_ylabel('変動率 (%)', fontsize=12, fontweight='bold')
        ax2.set_title('日次変動率', fontsize=14, fontweight='bold', pad=10)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # === グラフ3: 出来高（実際にはボラティリティ） ===
        ax3 = axes[2]

        # ボラティリティ（20日間の標準偏差）
        volatility = data['USD_JPY'].pct_change().rolling(window=20).std() * 100 * np.sqrt(252)
        ax3.plot(data.index, volatility, label='ボラティリティ(年率)', color='#8B4513', linewidth=2)
        ax3.fill_between(data.index, volatility, alpha=0.3, color='#8B4513')
        ax3.set_ylabel('ボラティリティ (%)', fontsize=12, fontweight='bold')
        ax3.set_xlabel('日付', fontsize=12, fontweight='bold')
        ax3.set_title('ボラティリティ推移', fontsize=14, fontweight='bold', pad=10)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax3.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # レイアウト調整
        plt.tight_layout()

        # 保存
        output_file = 'outputs/usd_jpy_price_chart.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\nグラフ保存: {output_file}")

        # 統計情報表示
        print("\n" + "=" * 80)
        print("統計情報")
        print("=" * 80)

        latest_7_days = data['USD_JPY'].tail(7)
        latest_30_days = data['USD_JPY'].tail(30)

        print(f"\n現在価格: {data['USD_JPY'].iloc[-1]:.2f}円")
        print(f"\n7日間:")
        print(f"  最高値: {latest_7_days.max():.2f}円")
        print(f"  最安値: {latest_7_days.min():.2f}円")
        print(f"  変動率: {((latest_7_days.iloc[-1] / latest_7_days.iloc[0]) - 1) * 100:+.2f}%")

        print(f"\n30日間:")
        print(f"  最高値: {latest_30_days.max():.2f}円")
        print(f"  最安値: {latest_30_days.min():.2f}円")
        print(f"  変動率: {((latest_30_days.iloc[-1] / latest_30_days.iloc[0]) - 1) * 100:+.2f}%")

        print(f"\n全期間 ({len(data)}日):")
        print(f"  最高値: {data['USD_JPY'].max():.2f}円 ({data['USD_JPY'].idxmax().strftime('%Y-%m-%d')})")
        print(f"  最安値: {data['USD_JPY'].min():.2f}円 ({data['USD_JPY'].idxmin().strftime('%Y-%m-%d')})")
        print(f"  平均値: {data['USD_JPY'].mean():.2f}円")
        print(f"  標準偏差: {data['USD_JPY'].std():.2f}円")

        current_volatility = volatility.iloc[-1]
        print(f"\n現在のボラティリティ: {current_volatility:.2f}% (年率)")

        # トレンド分析
        ma_7_latest = data['MA_7'].iloc[-1]
        ma_25_latest = data['MA_25'].iloc[-1]
        ma_50_latest = data['MA_50'].iloc[-1]

        print(f"\n移動平均:")
        print(f"  7日MA:  {ma_7_latest:.2f}円")
        print(f"  25日MA: {ma_25_latest:.2f}円")
        print(f"  50日MA: {ma_50_latest:.2f}円")

        print(f"\nトレンド分析:")
        if latest_price > ma_7_latest > ma_25_latest:
            print("  短期トレンド: 上昇")
        elif latest_price < ma_7_latest < ma_25_latest:
            print("  短期トレンド: 下降")
        else:
            print("  短期トレンド: 横ばい")

        if ma_7_latest > ma_50_latest:
            print("  中期トレンド: 上昇")
        elif ma_7_latest < ma_50_latest:
            print("  中期トレンド: 下降")
        else:
            print("  中期トレンド: 横ばい")

        print("\n" + "=" * 80)
        print(f"グラフを表示しています: {output_file}")
        print("=" * 80)

        # 表示
        plt.show()

    except Exception as e:
        print(f"\nエラー: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    create_price_chart()
