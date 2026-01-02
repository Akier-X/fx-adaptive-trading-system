"""
センチメント分析モジュール (FinBERT)
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict, Union
from loguru import logger


class SentimentAnalyzer:
    """
    金融ニュースのセンチメント分析

    FinBERT を使用して、ニュース記事の感情を分析
    - Positive (ポジティブ): 1
    - Neutral (中立): 0
    - Negative (ネガティブ): -1
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Args:
            model_name: 使用するモデル名 (デフォルト: FinBERT)
        """
        logger.info(f"センチメント分析モデルを読み込み中: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()

            # GPU利用可能なら使用
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)

            logger.info(f"モデル読み込み完了 (device: {self.device})")

        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
            raise

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        単一テキストのセンチメント分析

        Args:
            text: 分析するテキスト

        Returns:
            {
                'positive': float,
                'neutral': float,
                'negative': float,
                'sentiment_score': float (-1 ~ 1),
                'label': str
            }
        """
        if not text or len(text.strip()) == 0:
            return {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'sentiment_score': 0.0,
                'label': 'neutral'
            }

        try:
            # トークン化
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 予測
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predictions = predictions.cpu().numpy()[0]

            # FinBERT の出力: [positive, negative, neutral]
            positive = float(predictions[0])
            negative = float(predictions[1])
            neutral = float(predictions[2])

            # センチメントスコア計算 (-1 ~ 1)
            sentiment_score = positive - negative

            # ラベル
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'

            return {
                'positive': positive,
                'neutral': neutral,
                'negative': negative,
                'sentiment_score': sentiment_score,
                'label': label
            }

        except Exception as e:
            logger.error(f"センチメント分析エラー: {e}")
            return {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0,
                'sentiment_score': 0.0,
                'label': 'neutral'
            }

    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        複数テキストのバッチ処理

        Args:
            texts: テキストのリスト
            batch_size: バッチサイズ

        Returns:
            センチメント結果のリスト
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            for text in batch:
                result = self.analyze_text(text)
                results.append(result)

        return results

    def aggregate_sentiment(
        self,
        texts: List[str],
        weights: List[float] = None
    ) -> Dict[str, float]:
        """
        複数テキストのセンチメントを集約

        Args:
            texts: テキストのリスト
            weights: 各テキストの重み (オプション)

        Returns:
            集約されたセンチメント
        """
        if not texts:
            return {
                'avg_sentiment': 0.0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0,
                'count': 0
            }

        results = self.analyze_batch(texts)

        if weights is None:
            weights = [1.0] * len(results)

        total_weight = sum(weights)

        # 重み付き平均
        avg_sentiment = sum(
            r['sentiment_score'] * w
            for r, w in zip(results, weights)
        ) / total_weight

        # ラベルごとの割合
        positive_count = sum(1 for r in results if r['label'] == 'positive')
        negative_count = sum(1 for r in results if r['label'] == 'negative')
        neutral_count = sum(1 for r in results if r['label'] == 'neutral')

        total = len(results)

        return {
            'avg_sentiment': avg_sentiment,
            'positive_ratio': positive_count / total,
            'negative_ratio': negative_count / total,
            'neutral_ratio': neutral_count / total,
            'count': total,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count
        }

    def analyze_news_articles(self, articles: List[Dict]) -> Dict:
        """
        ニュース記事のセンチメント分析

        Args:
            articles: NewsCollector から取得した記事リスト

        Returns:
            分析結果
        """
        texts = []
        weights = []

        for article in articles:
            # タイトルと説明を結合
            text = f"{article.get('title', '')} {article.get('description', '')}"
            texts.append(text)

            # Redditの場合はスコアを重みとして使用
            if article.get('source') == 'reddit':
                weight = max(1, article.get('score', 1))
            else:
                weight = 1.0

            weights.append(weight)

        aggregated = self.aggregate_sentiment(texts, weights)

        # 個別の分析結果も保存
        individual_results = self.analyze_batch(texts)

        return {
            'aggregated': aggregated,
            'individual': individual_results,
            'total_articles': len(articles)
        }
