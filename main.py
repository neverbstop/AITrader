# main.py
from pipelines.data_pipeline import DataPipeline
from pipelines.news_pipeline import NewsPipeline
from pipelines.sentiment_pipeline import SentimentPipeline
from dashboard.dashboard import open_dashboard
import polars as pl
import config


# Configure Polars to use ASCII characters for table formatting.
# This prevents UnicodeEncodeError on some Windows terminals.
pl.Config.set_tbl_formatting("ASCII_FULL")

if __name__ == "__main__":
    print("Starting Trading AI Project...")

    # === Stock Data Pipeline ===
    data_pipeline = DataPipeline(stock_file=config.STOCK_FILE, ticker=config.TICKER)
    stock_data = data_pipeline.run()
    print("Stock Data Sample:")
    print(stock_data.head())

    # === News Pipeline ===
    news_pipeline = NewsPipeline(api_key=config.NEWS_API_KEY)
    # Use the Ticker from config, removing the exchange suffix for better news search results
    news_data = news_pipeline.run(company=config.TICKER.split('.')[0], days=3)
    print(f"\nCollected {len(news_data)} news articles.")

    # === Sentiment Pipeline ===
    sentiment_pipeline = SentimentPipeline(news_articles=news_data)
    sentiment_results = sentiment_pipeline.run()
    print("\nSentiment Sample:", sentiment_results[:2])

    # === Dashboard ===
    print("\nOpening dashboard...")
    open_dashboard()
