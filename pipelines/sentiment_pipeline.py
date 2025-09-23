# pipelines/sentiment_pipeline.py
from textblob import TextBlob

class SentimentPipeline:
    """
    Runs sentiment analysis on preprocessed news.
    """

    def __init__(self, news_articles: list):
        self.news_articles = news_articles
        self.results = []

    def analyze_sentiment(self, text: str) -> float:
        """Return polarity score between -1 and 1"""
        return TextBlob(text).sentiment.polarity

    def run(self):
        """Run sentiment analysis on all articles"""
        for article in self.news_articles:
            text = f"{article['title']} {article['description']} {article['content']}"
            sentiment = self.analyze_sentiment(text)
            article["sentiment"] = sentiment
            self.results.append(article)
        return self.results
