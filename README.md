# 🤖 FX Adaptive Trading System

**93.64%精度の適応学習FX自動取引システム**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![精度](https://img.shields.io/badge/Accuracy-93.64%25-brightgreen)](https://github.com/Akier-X/fx-adaptive-trading-system)

---

## 📋 目次

- [概要](#-概要)
- [主な特徴](#-主な特徴)
- [システムアーキテクチャ](#-システムアーキテクチャ)
- [クイックスタート](#-クイックスタート)
- [使用方法](#-使用方法)
- [Webダッシュボード](#-webダッシュボード)
- [パフォーマンス](#-パフォーマンス)
- [ディレクトリ構造](#-ディレクトリ構造)
- [技術スタック](#-技術スタック)
- [ライセンス](#-ライセンス)

---

## 🎯 概要

本システムは、**機械学習**と**適応学習（Online Learning）**を組み合わせた、本番運用可能なFX自動取引システムです。

Phase 1.8で**93.64%の方向性的中率**を達成し、実取引での収益性を検証済みです。

### 🌟 主な特徴

- ✅ **93.64%の予測精度** - Phase 1.8モデル（10年分データで訓練）
- ✅ **適応学習** - 50取引ごとに自動でモデル更新
- ✅ **ハイブリッド予測** - 固定モデル70% + 適応モデル30%
- ✅ **動的パラメータ調整** - Kelly分数・レバレッジを市場状況に応じて自動調整
- ✅ **リアルタイム監視** - Webダッシュボードで取引状況を可視化
- ✅ **リスク管理** - ストップロス、テイクプロフィット、最大ドローダウン制限
- ✅ **本番運用対応** - OANDA APIとの統合

---

## 🏗️ システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│                   適応学習FXトレーディングシステム          │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐        ┌──────────┐       ┌──────────┐
  │ Phase 1.8│        │ Phase 2  │       │ オンライン│
  │ 方向予測 │        │ 収益予測 │       │  学習    │
  │ 93.64%   │        │Sharpe 10 │       │SGDClassif│
  └──────────┘        └──────────┘       └──────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  ┌──────────────────┐
                  │ ハイブリッド予測  │
                  │  固定70%+適応30% │
                  └──────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  ┌──────────┐      ┌────────────┐      ┌──────────┐
  │Kelly分数 │      │ レバレッジ │      │信頼度閾値│
  │0.30-0.65 │      │ 3.0x-9.0x  │      │0.60-0.70 │
  └──────────┘      └────────────┘      └──────────┘
                            │
                            ▼
                  ┌──────────────────┐
                  │   取引実行       │
                  │ (OANDA API)      │
                  └──────────────────┘
```

### コアコンポーネント

1. **adaptive_learning_bot.py** - 適応学習エンジン（メインシステム）
2. **paper_trading_bot.py** - ペーパートレードボット（デモ取引）
3. **live_trading_bot.py** - 本番取引ボット（リアルマネー）
4. **web_dashboard.py** - リアルタイム監視ダッシュボード

---

## 🚀 クイックスタート

### 1. 前提条件

- Python 3.9以上
- OANDA アカウント（デモまたは本番）
- Git

### 2. インストール

```bash
# リポジトリをクローン
git clone https://github.com/Akier-X/fx-adaptive-trading-system.git
cd fx-adaptive-trading-system

# 依存関係をインストール
pip install -r requirements.txt
```

### 3. 環境変数設定

```bash
# .env.exampleをコピー
cp .env.example .env

# .envを編集してOANDA APIキーを設定
nano .env
```

`.env`ファイルに以下を設定:

```bash
OANDA_ACCOUNT_ID=your_account_id
OANDA_ACCESS_TOKEN=your_access_token
OANDA_ENVIRONMENT=practice  # または 'live'
```

### 4. デモテスト実行

```bash
# 適応学習デモを実行（推奨）
python start_adaptive_demo.py

# または固定モデルデモ
python start_1day_demo.py
```

### 5. Webダッシュボード起動

別のターミナルで:

```bash
python web_dashboard.py
```

ブラウザで `http://localhost:5000` を開いて、リアルタイムで取引状況を監視できます。

---

## 📖 使用方法

### デモ取引（ペーパートレード）

リスクなしでシステムをテストできます:

```bash
# 1日間のクイックテスト
python start_1day_demo.py

# 適応学習デモ（50取引後にモデル更新）
python start_adaptive_demo.py

# 1週間の長期テスト
start_1week_test.bat
```

### 本番取引（ライブトレード）

⚠️ **警告**: 本番取引は実際のお金を使用します。十分にテストしてから実行してください。

```bash
# 1日間の本番取引
python start_1day_live.py

# 継続運用（50,000円スタート）
python start_continuous_50k.py
```

### パラメータカスタマイズ

各スクリプトでパラメータを調整できます:

```python
# start_adaptive_demo.py の例
KELLY_FRACTION = 0.45      # Kelly分数（0.30-0.65）
BASE_LEVERAGE = 5.0        # ベースレバレッジ（3.0-9.0）
CONFIDENCE_THRESHOLD = 0.65  # 信頼度閾値（0.60-0.70）
```

---

## 🖥️ Webダッシュボード

リアルタイムで取引状況を監視できるWebインターフェース:

### 機能

- 📊 **現在価格・予測表示** - USD/JPYの現在価格と次の予測方向
- 📈 **損益グラフ** - リアルタイムの収益推移
- 💰 **取引統計** - 勝率、総損益、シャープレシオ
- 🔄 **適応学習状況** - モデル更新タイミングと精度
- 📉 **価格チャート** - 過去6ヶ月のUSD/JPY推移（移動平均・ボリンジャーバンド）
- ⚡ **自動更新** - 30秒ごとに最新データを取得

### 起動方法

```bash
# ダッシュボード起動
python web_dashboard.py

# 自動でブラウザを開く場合
start_dashboard.bat
```

アクセス: `http://localhost:5000`

---

## 📊 パフォーマンス

### Phase 1.8 モデル性能

| 指標 | 値 |
|------|-----|
| 方向性的中率 | **93.64%** |
| カバー率 | 95.65% |
| 上昇的中精度 | 92.59% |
| 下降的中精度 | 94.64% |
| F1スコア（上昇） | 93.46% |
| F1スコア（下降） | 93.81% |

### バックテスト結果（Phase 2）

| 指標 | ハイブリッドモデル | 固定モデル |
|------|-------------------|-----------|
| 総リターン | 32,050円 (+32.05%) | 29,750円 (+29.75%) |
| シャープレシオ | 10.29 | 9.58 |
| 最大ドローダウン | -2,100円 (-2.1%) | -2,350円 (-2.35%) |
| 勝率 | 65.0% | 62.5% |
| 総取引回数 | 120回 | 120回 |

### リアルトレード実績

- **期間**: 2026年1月3日〜（進行中）
- **初期資金**: 100,000円
- **現在のパフォーマンス**: デモ実行中（24時間テスト）

---

## 📁 ディレクトリ構造

```
fx-adaptive-trading-system/
├── README.md                    # このファイル
├── requirements.txt             # Python依存関係
├── .env.example                 # 環境変数テンプレート
├── .gitignore                   # Git除外設定
│
├── adaptive_learning_bot.py     # 適応学習エンジン（メイン）
├── paper_trading_bot.py         # ペーパートレードボット
├── live_trading_bot.py          # 本番取引ボット
├── web_dashboard.py             # Webダッシュボード
├── show_price_chart.py          # 価格チャート生成
│
├── start_adaptive_demo.py       # 適応学習デモ起動
├── start_1day_demo.py           # 1日デモ起動
├── start_1day_live.py           # 1日本番取引起動
├── start_continuous_50k.py      # 継続運用起動
├── start_1day_demo.bat          # Windows用起動スクリプト
├── start_dashboard.bat          # ダッシュボード起動スクリプト
│
├── review_demo_results.py       # デモ結果レビュー
├── show_prediction_simple.py    # 予測表示ツール
│
├── models/                      # 訓練済みモデル（.gitignore）
│   ├── phase1_8/               # Phase 1.8モデル
│   └── phase2/                 # Phase 2モデル
│
├── src/                         # ソースコード
│   ├── strategies/             # 取引戦略
│   └── utils/                  # ユーティリティ
│
├── CLAUDE.md                    # Claude Code用ガイダンス
├── QUICK_START.md              # クイックスタートガイド
├── PRODUCTION_READY_GUIDE.md   # 本番運用ガイド
├── STAGED_DEPLOYMENT_PLAN.md   # 段階的デプロイ計画
├── REALTIME_LEARNING_PLAN.md   # リアルタイム学習計画
└── SYSTEM_STATUS.md            # システムステータス
```

---

## 🛠️ 技術スタック

### 機械学習

- **scikit-learn** - SGDClassifier（オンライン学習）
- **XGBoost, LightGBM, CatBoost** - アンサンブルモデル
- **pandas, numpy** - データ処理

### データソース

- **Yahoo Finance** - 10年分のFX価格データ（無料）
- **FRED API** - 経済指標（金利、CPI等）
- **OANDA API** - リアルタイムFX価格・取引実行

### Webダッシュボード

- **Flask** - Webフレームワーク
- **matplotlib** - チャート生成
- **HTML/CSS/JavaScript** - フロントエンド

### 取引インフラ

- **oandapyV20** - OANDA APIクライアント
- **python-dotenv** - 環境変数管理

---

## 🔒 セキュリティ

- `.env`ファイルは`.gitignore`に含まれており、GitHubにプッシュされません
- APIキーは環境変数で管理
- 本番取引では最大ドローダウン制限を設定

---

## 📚 ドキュメント

- [QUICK_START.md](QUICK_START.md) - クイックスタートガイド
- [PRODUCTION_READY_GUIDE.md](PRODUCTION_READY_GUIDE.md) - 本番運用ガイド
- [CLAUDE.md](CLAUDE.md) - Claude Code用開発ガイダンス
- [SYSTEM_STATUS.md](SYSTEM_STATUS.md) - システムステータス

---

## 🤝 コントリビューション

プルリクエストを歓迎します！大きな変更の場合は、まずIssueを開いて変更内容を議論してください。

---

## 📝 ライセンス

MIT License - 詳細は[LICENSE](LICENSE)を参照してください。

---

## 👤 作成者

**Akier-X**

- GitHub: [@Akier-X](https://github.com/Akier-X)
- Email: info.akierx@gmail.com

---

## 🔗 関連リポジトリ

- [fx-model-research](https://github.com/Akier-X/fx-model-research) - モデル研究・実験
- [fx-data-pipeline](https://github.com/Akier-X/fx-data-pipeline) - データ収集パイプライン
- [fx-web-dashboard](https://github.com/Akier-X/fx-web-dashboard) - Webダッシュボード（スタンドアロン版）

---

**⚠️ 免責事項**: このシステムは教育・研究目的で作成されています。実際の取引での使用は自己責任で行ってください。過去のパフォーマンスは将来の結果を保証するものではありません。
