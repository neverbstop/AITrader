import os
import requests
import pandas as pd
import yfinance as yf
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
NEWS_API_KEY = os.getenv("cf7e675c42814679abe01317c33a4cf4")

BASE_URL = "https://newsapi.org/v2/everything"


# -------------------- NEWS --------------------
def fetch_news(query="AAPL", from_days=7, language="en", page_size=50):
    """
    Fetch news from NewsAPI for a specific stock/company.
    """
    from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "language": language,
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    if response.status_code != 200:
        raise Exception(f"Error fetching news: {response.text}")
    return response.json().get("articles", [])


def preprocess_news(articles):
    """
    Converts raw articles to structured DataFrame with sentiment score.
    """
    data = []
    for art in articles:
        title = art.get("title", "")
        description = art.get("description", "")
        content = art.get("content", "")
        published = art.get("publishedAt", "")

        # Clean text
        text = " ".join([title or "", description or "", content or ""])
        sentiment = TextBlob(text).sentiment.polarity  # -1 to +1

        data.append({
            "title": title,
            "description": description,
            "publishedAt": published,
            "sentiment": sentiment
        })
    df = pd.DataFrame(data)

    if not df.empty:
        df["publishedAt"] = pd.to_datetime(df["publishedAt"])
        df["date"] = df["publishedAt"].dt.date
    return df


# -------------------- STOCKS --------------------
def fetch_stock_data(ticker="AAPL", from_days=7):
    """
    Fetch stock OHLCV data from Yahoo Finance.
    """
    start = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    stock = yf.download(ticker, start=start, end=end, interval="1d")
    stock.reset_index(inplace=True)
    stock["date"] = stock["Date"].dt.date
    return stock[["date", "Open", "High", "Low", "Close", "Volume"]]


# -------------------- MERGE --------------------
def merge_news_stocks(news_df, stock_df):
    """
    Merge news sentiment with stock data by date.
    """
    if news_df.empty:
        print("⚠️ No news data to merge.")
        return stock_df

    sentiment_daily = news_df.groupby("date")["sentiment"].mean().reset_index()
    merged = pd.merge(stock_df, sentiment_daily, on="date", how="left")
    return merged


# -------------------- SAVE --------------------
def save_pipeline(df, filename="data/news_stock_sentiment.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Saved merged dataset to {filename}")


# -------------------- MAIN --------------------
if __name__ == "__main__":
    company = "AAPL"  # Example: Apple
    print(f"🔎 Fetching news & stock data for {company}...")

    articles = fetch_news(query=company, from_days=7)
    news_df = preprocess_news(articles)
    stock_df = fetch_stock_data(ticker=company, from_days=7)

    merged_df = merge_news_stocks(news_df, stock_df)
    save_pipeline(merged_df)

    print(merged_df.head())
