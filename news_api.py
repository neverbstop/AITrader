# news_api.py
import requests
import pandas as pd
from datetime import datetime, timedelta

# ----------------------------------------
# CONFIG
# ----------------------------------------
API_KEY = "YOUR_NEWSAPI_KEY"   # replace with your NewsAPI key
COMPANY = "Apple"              # you can change this dynamically later
FROM_DAYS = 7                  # number of days back to fetch news
OUTPUT_FILE = f"data/news_{COMPANY.lower()}.csv"

# ----------------------------------------
# FETCH NEWS
# ----------------------------------------
def fetch_news(company=COMPANY, days=FROM_DAYS):
    url = "https://newsapi.org/v2/everything"
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    params = {
        "q": company,
        "from": from_date,
        "to": to_date,
        "sortBy": "publishedAt",
        "language": "en",
        "apiKey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if response.status_code != 200 or "articles" not in data:
        print("Error fetching news:", data)
        return pd.DataFrame()

    articles = data["articles"]
    news_list = []
    for article in articles:
        news_list.append({
            "publishedAt": article["publishedAt"],
            "source": article["source"]["name"],
            "title": article["title"],
            "description": article.get("description", ""),
            "url": article["url"]
        })

    df = pd.DataFrame(news_list)
    return df

# ----------------------------------------
# SAVE TO CSV
# ----------------------------------------
def save_news(df, filename=OUTPUT_FILE):
    if df.empty:
        print("No news fetched. Nothing to save.")
        return

    df["publishedAt"] = pd.to_datetime(df["publishedAt"])
    df.sort_values("publishedAt", inplace=True)

    # Append if file exists, else create new
    try:
        existing = pd.read_csv(filename)
        df = pd.concat([existing, df]).drop_duplicates(subset=["url"])
    except FileNotFoundError:
        pass

    df.to_csv(filename, index=False)
    print(f"✅ News saved to {filename}")

if __name__ == "__main__":
    df = fetch_news()
    save_news(df)
    print(df.head())
