# pipelines/news_pipeline.py
import requests
import datetime

class NewsPipeline:
    """
    Handles news collection, filtering, and preprocessing for Apple Inc.
    """

    def __init__(self, api_key: str, query: str = "Apple Inc. AAPL"):
        self.api_key = api_key
        self.query = query
        self.base_url = "https://newsapi.org/v2/everything"
        self.news_data = []

    def fetch_news(self, days: int = 7):
        """Fetch last N days of news related to Apple"""
        from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        params = {
            "q": self.query,
            "from": from_date,
            "sortBy": "relevancy",
            "apiKey": self.api_key,
            "language": "en"
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()

        if "articles" in data:
            self.news_data = data["articles"]
        return self.news_data

    def preprocess(self):
        """Clean up news fields into structured format"""
        return [
            {
                "date": article["publishedAt"],
                "source": article["source"]["name"],
                "title": article["title"],
                "description": article.get("description", ""),
                "content": article.get("content", "")
            }
            for article in self.news_data
        ]

    def run(self, days: int = 7):
        """Full pipeline: fetch → preprocess → return news list"""
        self.fetch_news(days=days)
        return self.preprocess()
