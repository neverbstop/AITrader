# make_dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------
# 1. FinBERT for encoding headlines
# ---------------------------------------------------
class FinBERTEncoder:
    def __init__(self, model_name="yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, texts, max_len=64):
        """Return CLS embeddings for a list of texts"""
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**tokens)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [batch, 768]
        return cls_embeddings.numpy()

# ---------------------------------------------------
# 2. Dataset wrapper
# ---------------------------------------------------
class TradingDataset(Dataset):
    def __init__(self, market_df, news_embeddings, Tm=60, Tn=5, H=10, threshold=0.005):
        """
        market_df: DataFrame with OHLCV, sorted by time
        news_embeddings: dict {timestamp: list of embeddings}
        """
        self.market_df = market_df.reset_index(drop=True)
        self.news_embeddings = news_embeddings
        self.Tm, self.Tn, self.H, self.threshold = Tm, Tn, H, threshold

    def __len__(self):
        return len(self.market_df) - self.Tm - self.H

    def __getitem__(self, idx):
        # ---- Market window ----
        market_seq = self.market_df.iloc[idx:idx+self.Tm][["open","high","low","close","volume"]].values
        market_seq = torch.tensor(market_seq, dtype=torch.float)

        # ---- News window ----
        t_now = self.market_df.iloc[idx+self.Tm-1]["timestamp"]
        # get last Tn news before this timestamp
        news_list = self.news_embeddings.get(t_now, [])
        if len(news_list) < self.Tn:
            # pad with zeros
            pad = [np.zeros(768)] * (self.Tn - len(news_list))
            news_list = pad + news_list
        else:
            news_list = news_list[-self.Tn:]
        news_seq = torch.tensor(news_list, dtype=torch.float)

        # ---- Label ----
        p_now = self.market_df.iloc[idx+self.Tm-1]["close"]
        p_future = self.market_df.iloc[idx+self.Tm-1+self.H]["close"]

        if p_future > p_now * (1 + self.threshold):
            label = 2  # Buy
        elif p_future < p_now * (1 - self.threshold):
            label = 0  # Sell
        else:
            label = 1  # Hold

        return market_seq, news_seq, torch.tensor(label)

# ---------------------------------------------------
# 3. Main preprocessing pipeline
# ---------------------------------------------------
def build_dataset(market_csv, news_csv, out_path="processed_dataset.pt",
                  Tm=60, Tn=5, H=10, threshold=0.005):
    # ---- Load OHLCV ----
    market_df = pd.read_csv(market_csv, parse_dates=["timestamp"])
    market_df = market_df.sort_values("timestamp")

    # ---- Load news ----
    news_df = pd.read_csv(news_csv, parse_dates=["timestamp"])
    news_df = news_df.sort_values("timestamp")

    encoder = FinBERTEncoder()
    news_embeddings = {}

    for _, row in news_df.iterrows():
        ts = row["timestamp"]
        headline = str(row["headline"])
        emb = encoder.encode([headline])[0]  # single vector
        # bucket by nearest market timestamp
        ts_key = market_df.iloc[(market_df["timestamp"]-ts).abs().argmin()]["timestamp"]
        news_embeddings.setdefault(ts_key, []).append(emb)

    # ---- Wrap into dataset ----
    dataset = TradingDataset(market_df, news_embeddings, Tm, Tn, H, threshold)

    # ---- Save ----
    torch.save(dataset, out_path)
    print(f"✅ Saved dataset to {out_path} with {len(dataset)} samples.")


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    build_dataset("data/raw/ohlcv.csv", "data/raw/news.csv",
                  out_path="data/processed/trading_dataset.pt")
