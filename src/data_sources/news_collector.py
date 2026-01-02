"""
ニュースデータ収集モジュール
"""
from newsapi import NewsApiClient
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd


class NewsCollector:
    """
    複数のソースからニュースを収集

    対応API:
    - NewsAPI: 世界中のニュース
    - Alpha Vantage: 金融ニュース
    - Reddit: コミュニティセンチメント
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None
    ):
        self.newsapi_key = newsapi_key
        self.alpha_vantage_key = alpha_vantage_key

        if newsapi_key:
            self.newsapi = NewsApiClient(api_key=newsapi_key)
        else:
            self.newsapi = None
            logger.warning("NewsAPI key not provided")

    def get_forex_news(
        self,
        query: str = "forex OR currency OR USD OR JPY",
        days_back: int = 1,
        language: str = "en"
    ) -> List[Dict]:
        """
        FX関連のニュースを取得

        Args:
            query: 検索クエリ
            days_back: 何日前までのニュースを取得するか
            language: 言語 (en, ja など)

        Returns:
            ニュース記事のリスト
        """
        if not self.newsapi:
            logger.warning("NewsAPI not initialized")
            return []

        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

            response = self.newsapi.get_everything(
                q=query,
                from_param=from_date,
                language=language,
                sort_by='publishedAt',
                page_size=100
            )

            articles = response.get('articles', [])
            logger.info(f"取得したニュース記事数: {len(articles)}")

            return articles

        except Exception as e:
            logger.error(f"ニュース取得エラー: {e}")
            return []

    def get_economic_calendar(self) -> List[Dict]:
        """
        経済指標カレンダーを取得

        Returns:
            経済指標イベントのリスト
        """
        # Alpha Vantage の経済カレンダー
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage key not provided")
            return []

        try:
            url = f"https://www.alphavantage.co/query"
            params = {
                "function": "ECONOMIC_CALENDAR",
                "apikey": self.alpha_vantage_key
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            logger.info(f"経済指標取得: {len(data)} イベント")

            return data

        except Exception as e:
            logger.error(f"経済指標取得エラー: {e}")
            return []

    def get_fear_greed_index(self) -> Optional[Dict]:
        """
        Fear & Greed Index を取得

        Returns:
            市場心理指数
        """
        try:
            # Alternative.me の Fear & Greed Index
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            data = response.json()
            if data and 'data' in data and len(data['data']) > 0:
                current = data['data'][0]
                logger.info(f"Fear & Greed Index: {current['value']} ({current['value_classification']})")
                return current

            return None

        except Exception as e:
            logger.error(f"Fear & Greed Index取得エラー: {e}")
            return None

    def get_reddit_sentiment(
        self,
        subreddit: str = "forex",
        limit: int = 100
    ) -> List[Dict]:
        """
        Redditの投稿を取得 (センチメント分析用)

        Args:
            subreddit: サブレディット名
            limit: 取得件数

        Returns:
            投稿のリスト
        """
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {'User-Agent': 'Mozilla/5.0'}

            response = requests.get(
                url,
                headers=headers,
                params={'limit': limit},
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            posts = []

            for child in data.get('data', {}).get('children', []):
                post_data = child.get('data', {})
                posts.append({
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'created': datetime.fromtimestamp(post_data.get('created_utc', 0)),
                    'url': post_data.get('url', ''),
                    'num_comments': post_data.get('num_comments', 0)
                })

            logger.info(f"Reddit投稿取得: {len(posts)}件 from r/{subreddit}")
            return posts

        except Exception as e:
            logger.error(f"Reddit取得エラー: {e}")
            return []

    def aggregate_news_sentiment(
        self,
        query: str = "USD JPY",
        days_back: int = 1
    ) -> Dict:
        """
        複数ソースからニュースを集約し、サマリーを作成

        Args:
            query: 検索クエリ
            days_back: 何日前まで

        Returns:
            集約されたニュースデータ
        """
        all_news = []

        # NewsAPI
        if self.newsapi:
            news_articles = self.get_forex_news(query, days_back)
            for article in news_articles:
                all_news.append({
                    'source': 'newsapi',
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'published_at': article.get('publishedAt', ''),
                    'url': article.get('url', '')
                })

        # Reddit
        reddit_posts = self.get_reddit_sentiment('forex', limit=50)
        for post in reddit_posts:
            all_news.append({
                'source': 'reddit',
                'title': post['title'],
                'description': post['text'][:500] if post['text'] else '',
                'content': post['text'],
                'published_at': post['created'].isoformat(),
                'url': post['url'],
                'score': post['score']
            })

        # Fear & Greed
        fg_index = self.get_fear_greed_index()

        return {
            'news_count': len(all_news),
            'articles': all_news,
            'fear_greed_index': fg_index,
            'query': query,
            'collected_at': datetime.now().isoformat()
        }

    def save_to_dataframe(self, news_data: Dict) -> pd.DataFrame:
        """
        ニュースデータをDataFrameに変換

        Args:
            news_data: aggregate_news_sentiment の結果

        Returns:
            DataFrame
        """
        if not news_data.get('articles'):
            return pd.DataFrame()

        df = pd.DataFrame(news_data['articles'])
        df['published_at'] = pd.to_datetime(df['published_at'])
        df['collected_at'] = news_data['collected_at']

        if 'fear_greed_index' in news_data and news_data['fear_greed_index']:
            df['fear_greed_value'] = news_data['fear_greed_index']['value']
            df['fear_greed_class'] = news_data['fear_greed_index']['value_classification']

        return df
