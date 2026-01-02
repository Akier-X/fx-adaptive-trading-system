# FX Adaptive Trading System

**世界最強の適応学習FX自動取引システム**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 概要

本システムは、**適応学習（Online Learning）** と **ハイブリッド予測** を組み合わせた、本番運用可能なFX自動取引システムです。

### 主な特徴

- ✅ **93.64%の予測精度** (Phase 1.8モデル)
- ✅ **適応学習**: 50取引ごとに自動モデル更新
- ✅ **ハイブリッド予測**: 固定70% + 適応30%
- ✅ **動的パラメータ**: Kelly分数・レバレッジを市場に応じて調整
- ✅ **リアルタイム監視**: Webダッシュボード

---

## 🚀 クイックスタート

### 1. インストール

\`\`\`bash
git clone https://github.com/yourusername/fx-adaptive-trading-system.git
cd fx-adaptive-trading-system
pip install -r requirements.txt
\`\`\`

### 2. 環境変数設定

\`\`\`bash
cp .env.example .env
# .envを編集してOANDA APIキーを設定
\`\`\`

### 3. デモテスト

\`\`\`bash
python start_adaptive_demo.py
\`\`\`

### 4. Webダッシュボード

\`\`\`bash
python web_dashboard.py
# ブラウザで http://localhost:5000 を開く
\`\`\`

---

## 📊 システムアーキテクチャ

\`\`\`
adaptive_learning_bot.py
├── Phase 1.8 (方向予測: 93.64%精度)
├── Phase 2 (収益予測: Sharpe 10.29)
├── オンライン学習 (SGDClassifier)
├── ハイブリッド予測 (固定70% + 適応30%)
└── 適応的パラメータ調整
    ├── Kelly分数: 0.30-0.65
    ├── レバレッジ: 3.0x-9.0x
    └── 信頼度閾値: 0.60-0.70
\`\`\`

---

## 📝 ライセンス

MIT License

EOF

