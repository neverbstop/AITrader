import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification # Use a Hugging Face tokenizer for simplicity
from transformer_model import SentimentTransformer # Import our custom transformer model
from config import NEWS_API_KEY, CHECK_INTERVAL_SECONDS
import os

class NewsAgent:
    def __init__(self, ticker):
        self.ticker = ticker
        self.last_checked_time = datetime.now()

        # Load the custom-trained transformer model and tokenizer
        model_dir = "models"
        model_path = os.path.join(model_dir, "sentiment_transformer.pth")
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        
        if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
            raise FileNotFoundError("Trained sentiment model or tokenizer not found. Please run train_sentiment_model.py first.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Initialize and load the model
        vocab_size = self.tokenizer.vocab_size
        num_classes = 3
        self.model = SentimentTransformer(
            vocab_size=vocab_size,
            d_model=768,
            num_heads=12,
            d_ff=3072,
            num_layers=2,
            dropout=0.1,
            num_classes=num_classes
        )
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def fetch_recent_news(self, query, from_days=1):
        """
        Fetches recent news articles for a given query.
        """
        if not NEWS_API_KEY:
            print("⚠️ News API key not found. Please set it in your .env file.")
            return []

        url = "https://newsapi.org/v2/everything"
        from_date = (datetime.now() - timedelta(days=from_days)).strftime("%Y-%m-%d")
        
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 50,
            "apiKey": NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            return articles
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news from API: {e}")
            return []

    def analyze_sentiment(self, articles):
        """
        Analyzes the sentiment of news headlines and descriptions using the custom transformer model.
        """
        if not articles:
            return None, None

        texts = [f"{art.get('title', '')} {art.get('description', '')}" for art in articles]
        
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=128,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask)
            probabilities = torch.softmax(output, dim=1)

        # Map predictions to labels
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        predictions = [sentiment_map[torch.argmax(p).item()] for p in probabilities]

        # Calculate an average numerical score
        numerical_scores = []
        for prob in probabilities:
            if torch.argmax(prob).item() == 0:
                numerical_scores.append(-prob[0].item())  # Negative score
            elif torch.argmax(prob).item() == 2:
                numerical_scores.append(prob[2].item())   # Positive score
            else:
                numerical_scores.append(0)                # Neutral score

        average_sentiment = sum(numerical_scores) / len(numerical_scores) if numerical_scores else 0
        
        return average_sentiment, predictions

    def check_for_signals(self):
        """
        Checks for major news events and returns a signal if one is found.
        """
        articles = self.fetch_recent_news(self.ticker)
        if not articles:
            return None, None

        average_sentiment, predictions = self.analyze_sentiment(articles)
        
        if average_sentiment > 0.3:
            return f"Significant positive news detected! Average sentiment score: {average_sentiment:.2f}", predictions
        elif average_sentiment < -0.3:
            return f"Significant negative news detected! Average sentiment score: {average_sentiment:.2f}", predictions
            
        return None, None

if __name__ == "__main__":
    # This block allows you to run the agent independently for testing
    try:
        agent = NewsAgent(ticker="RELIANCE.NS")
        print("NewsAgent initialized with custom transformer model.")
        while True:
            signal, predictions = agent.check_for_signals()
            if signal:
                print(signal)
                if predictions:
                    # An example of how XAI could be integrated here would be to
                    # print the specific headlines that contributed to the extreme
                    # sentiment score.
                    print("Top sentiment predictions:")
                    for i, pred in enumerate(predictions):
                        print(f"  - {articles[i]['title']}: {pred}")
            else:
                print("No significant news detected. Monitoring...")
            time.sleep(CHECK_INTERVAL_SECONDS)
    except FileNotFoundError as e:
        print(e)
