# core/news_agent.py
import requests
import logging
from config import NEWS_API_KEY

class NewsAgent:
    def __init__(self):
        self.api_key = NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query="stock market", language="en", page_size=10):
        """Fetch latest financial news from NewsAPI."""
        try:
            url = f"{self.base_url}?q={query}&language={language}&pageSize={page_size}&apiKey={self.api_key}"
            response = requests.get(url)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            logging.info(f"📰 Retrieved {len(articles)} news articles.")
            return articles
        except Exception as e:
            logging.error(f"❌ Error fetching news: {e}")
            return []

    def process_news(self, sentiments):
        """
        Convert sentiment analysis into trading signals.
        Example: if sentiment is strongly positive → BUY signal.
        """
        signals = []
        for sentiment in sentiments:
            score = sentiment.get("score", 0)
            if score > 0.5:
                signals.append("BUY")
            elif score < -0.5:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        logging.info(f"🧠 News signals generated: {signals}")
        return signals
