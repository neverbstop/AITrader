# sentiment.py
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px

# Ensure VADER is available
nltk.download('vader_lexicon', quiet=True)

def analyze_sentiment(input_csv="news_data.csv", output_csv="news_with_sentiment.csv", plot_html="sentiment_trend.html"):
    """
    Reads news data from CSV, applies sentiment analysis, saves results,
    and plots daily sentiment trends.
    """
    # Step 1: Load news data
    df = pd.read_csv(input_csv)

    if df.empty:
        print("No news data found. Run news_api.py first.")
        return None

    # Step 2: Initialize VADER
    sia = SentimentIntensityAnalyzer()

    # Step 3: Apply sentiment to each news title + description
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    sentiments = df["text"].apply(sia.polarity_scores).apply(pd.Series)
    df = pd.concat([df, sentiments], axis=1)

    # Step 4: Create a final label based on compound score
    df["sentiment_label"] = df["compound"].apply(
        lambda x: "positive" if x > 0.05 else ("negative" if x < -0.05 else "neutral")
    )

    # Step 5: Save with sentiments
    df.to_csv(output_csv, index=False)
    print(f"Sentiment analysis completed. Results saved to {output_csv}")

    # Step 6: Convert publishedAt to datetime
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["date"] = df["publishedAt"].dt.date

    # Step 7: Aggregate daily sentiment (average compound score)
    daily_sentiment = df.groupby("date")["compound"].mean().reset_index()

    # Step 8: Plot with Plotly
    fig = px.line(
        daily_sentiment,
        x="date",
        y="compound",
        title="Daily Average News Sentiment",
        labels={"date": "Date", "compound": "Average Sentiment Score"},
        markers=True
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red")  # neutral baseline

    # Save interactive dashboard as HTML
    fig.write_html(plot_html)
    print(f"Sentiment trend chart saved as {plot_html}")

    return df, daily_sentiment


if __name__ == "__main__":
    analyze_sentiment()
