# クイックスタートガイド

## 🚀 今すぐ始める

### 1. ペーパートレーディング（仮想資金テスト）

**最も簡単な方法（Windows）**:
```
start_1week_test.bat をダブルクリック
```

**Pythonコマンド**:
```bash
# 1時間クイックテスト（推奨 - 最初にこれを実行）
python start_quick_test.py

# 7日間フルテスト
python start_1week_test.py

# 単発実行（テスト用）
python paper_trading_bot.py
```

---

## 📊 結果確認

### リアルタイム監視

```bash
# 10秒ごとに自動更新
python monitor_test.py
```

### 最新結果表示

```bash
python view_results.py
```

### ログファイル確認

```bash
# Windows PowerShell
Get-Content logs\1week_test_*.log -Wait

# Linux/Mac
tail -f logs/1week_test_*.log
```

---

## 🖥️ サーバー移行

### 自動セットアップ（Ubuntu/Debian）

```bash
# 1. サーバーにスクリプトをコピー
scp server_setup.sh user@server-ip:/tmp/

# 2. サーバー上で実行
ssh user@server-ip
chmod +x /tmp/server_setup.sh
/tmp/server_setup.sh

# 3. システムファイルを転送
cd D:\FX
scp -r * user@server-ip:/opt/fx-bot/

# 4. サーバー上でサービス起動
ssh user@server-ip
sudo systemctl enable fx-bot
sudo systemctl start fx-bot
sudo systemctl status fx-bot
```

---

## 📁 ファイル構成

### 実行スクリプト

| ファイル | 用途 |
|---------|------|
| `start_1week_test.bat` | Windows用ワンクリック起動 |
| `start_1week_test.py` | 7日間テスト（Python） |
| `start_quick_test.py` | 1時間クイックテスト |
| `paper_trading_bot.py` | ペーパートレーディング本体 |
| `live_trading_bot.py` | 本番取引ボット（OANDA API） |

### ユーティリティ

| ファイル | 用途 |
|---------|------|
| `view_results.py` | 最新結果表示 |
| `monitor_test.py` | リアルタイム監視 |
| `server_setup.sh` | サーバー自動セットアップ |

### ドキュメント

| ファイル | 内容 |
|---------|------|
| `DEPLOYMENT_GUIDE.md` | 詳細デプロイメントガイド |
| `QUICK_START.md` | このファイル |
| `FINAL_SYSTEM_REPORT.md` | システム完成レポート |

---

## ⚙️ 設定ファイル

### .env（環境変数）

```bash
# OANDA API
OANDA_ACCOUNT_ID=001-009-20044456-001
OANDA_ACCESS_TOKEN=your_token_here
OANDA_ENVIRONMENT=practice  # デモ口座（安全）

# 初期資金
INITIAL_CAPITAL=10000

# リスク設定（本番用は小さく）
MAX_RISK_PER_TRADE=0.02
LEVERAGE=10  # 本番は2.5推奨
```

---

## 📈 期待される結果

### 7日間テスト

バックテスト実績から推定:

| 指標 | 推定値 |
|------|--------|
| 初期資金 | ¥10,000 |
| 最終資金 | ¥10,570 |
| リターン | +5.7% |
| 取引回数 | 3-5回 |
| 勝率 | 90%+ |
| 最大DD | <5% |

---

## 🆘 トラブルシューティング

### モデルファイルが見つからない

```bash
# モデルを訓練
python train_and_save_models.py
```

### OANDA API認証エラー

1. `.env`ファイルでトークン確認
2. OANDAサイトでトークン再発行
3. `OANDA_ENVIRONMENT=practice`を確認

### Yahoo Financeデータ取得失敗

- インターネット接続確認
- Yahoo Financeサービス状況確認

---

## 🎯 実行フロー

```
1. クイックテスト（1時間）
   ↓ 成功
2. 7日間テスト（このPC）
   ↓ 成功
3. サーバー移行
   ↓
4. サーバーで7日間テスト
   ↓ 成功
5. 本番運用開始（少額）
```

---

## 📞 重要な注意事項

### ⚠️ 本番運用前に

1. **パラメータ調整**:
   - Kelly: 0.70 → 0.30-0.40（保守的に）
   - レバレッジ: 10x → 2.5-5x

2. **少額スタート**:
   - 初回は1-3万円推奨
   - 成功したら徐々に増額

3. **監視体制**:
   - 日次でログ確認
   - 異常時のアラート設定

---

**作成日**: 2026-01-03
**バージョン**: 1.0
