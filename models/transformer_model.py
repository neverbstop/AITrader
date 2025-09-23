"""
models/transformer_model.py

Custom multimodal Transformer skeleton for trading:
- Market encoder: encodes sequences of numeric market features (OHLCV + indicators)
- News encoder: accepts precomputed text embeddings (or raw text embedding step can be added externally)
- Fusion: combines market tokens + news tokens and feeds into TransformerEncoder
- Head: classification head -> [Buy, Hold, Sell]

Usage:
    from models.transformer_model import MultimodalTransformer, make_dummy_input
    model = MultimodalTransformer(...)
    market_x, news_x = make_dummy_input(...)
    logits = model(market_x, news_x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequences."""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        if d_model % 2 == 1:
            # odd dim handling
            pe[:, 1::2] = torch.cos(pos * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        length = x.size(1)
        return x + self.pe[:, :length, :].to(x.dtype)


class MarketEncoder(nn.Module):
    """
    Encodes numeric market features into token embeddings.
    Input: market_x [B, T, F] where F = number of numeric features per timestep
    Output: market_tokens [B, T, d_model]
    """
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, F] float tensor
        returns: [B, T, d_model]
        """
        x = self.proj(x)              # [B, T, d_model]
        x = self.pos_enc(x)          # add positional info
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class NewsEncoder(nn.Module):
    """
    Encodes news embeddings into tokens that can be fused.
    Two modes:
      - per-timestep news embeddings: news_x [B, T_news, emb_dim]
      - single-vector per timestep (we repeat or align externally)
    This module projects embedding dimension to d_model, adds pos enc optionally.
    """
    def __init__(self, emb_dim: int, d_model: int, use_pos: bool = False, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(emb_dim, d_model)
        self.use_pos = use_pos
        self.pos_enc = PositionalEncoding(d_model) if use_pos else None
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, news_emb: torch.Tensor) -> torch.Tensor:
        """
        news_emb: [B, Tn, emb_dim] or [B, emb_dim] for a single vector per sample
        returns: [B, Tn, d_model] or [B, 1, d_model]
        """
        if news_emb.dim() == 2:
            # [B, emb_dim] -> [B, 1, emb_dim]
            news_emb = news_emb.unsqueeze(1)
        x = self.proj(news_emb)
        if self.use_pos:
            x = self.pos_enc(x)
        x = self.layernorm(x)
        x = self.dropout(x)
        return x


class FusionTransformer(nn.Module):
    """
    Fusion block: a stack of TransformerEncoder layers operating on concatenated tokens.
    """
    def __init__(self, d_model: int, n_heads: int = 8, n_layers: int = 4, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward, dropout, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        tokens: [S, B, d_model] when using nn.Transformer (seq-first)
        returns: [S, B, d_model]
        """
        out = self.transformer(tokens, src_key_padding_mask=src_key_padding_mask)
        return out


class MultimodalTransformer(nn.Module):
    """
    Full multimodal model:
      - market encoder -> tokens (T_market)
      - news encoder -> tokens (T_news) (can be length 1 or >1)
      - concatenate -> run through FusionTransformer
      - pooling + classification head -> buy/hold/sell logits
    """
    def __init__(
        self,
        n_market_features: int,
        news_emb_dim: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 3,
        pool_method: str = "cls"  # "cls" or "mean"
    ):
        super().__init__()
        self.market_encoder = MarketEncoder(n_market_features, d_model, dropout)
        self.news_encoder = NewsEncoder(news_emb_dim, d_model, use_pos=False, dropout=dropout)

        self.fusion = FusionTransformer(d_model, n_heads, n_layers, dim_feedforward, dropout)
        self.pool_method = pool_method

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        # A learnable CLS token to pool sequence if pool_method == "cls"
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, market_x: torch.Tensor, news_x: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        market_x: [B, Tm, F]
        news_x:  None or [B, Tn, emb_dim] or [B, emb_dim]
        returns: logits [B, num_classes]
        """
        B = market_x.size(0)
        market_tokens = self.market_encoder(market_x)   # [B, Tm, d]
        if news_x is None:
            # if no news, use a zero vector as single token
            device = market_tokens.device
            news_tokens = torch.zeros(B, 1, market_tokens.size(-1), device=device)
        else:
            news_tokens = self.news_encoder(news_x)     # [B, Tn, d]

        # Concatenate: [B, 1 (cls) + Tm + Tn, d]
        cls = self.cls_token.expand(B, -1, -1)
        concat = torch.cat([cls, market_tokens, news_tokens], dim=1)

        # Transformer in PyTorch expects seq-first: [S, B, d]
        tokens = concat.transpose(0, 1)  # [S, B, d]

        if attn_mask is not None:
            # attn_mask expected in torch transformer as src_key_padding_mask: [B, S]
            src_key_padding_mask = attn_mask
        else:
            src_key_padding_mask = None

        transformed = self.fusion(tokens, src_key_padding_mask=src_key_padding_mask)  # [S, B, d]
        transformed = transformed.transpose(0, 1)  # [B, S, d]

        if self.pool_method == "cls":
            pooled = transformed[:, 0, :]  # CLS token output
        else:
            pooled = transformed.mean(dim=1)

        logits = self.classifier(pooled)  # [B, num_classes]
        return logits

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, map_location="cpu"):
        self.load_state_dict(torch.load(path, map_location=map_location))


# --------------------------
# Utility: create dummy inputs
# --------------------------
def make_dummy_input(batch=2, Tm=60, n_features=6, Tn=3, emb_dim=768, device="cpu"):
    """
    Returns (market_x, news_x) on device for quick smoke test.
    market_x: [B, Tm, n_features]
    news_x: [B, Tn, emb_dim]
    """
    market_x = torch.randn(batch, Tm, n_features, device=device)
    news_x = torch.randn(batch, Tn, emb_dim, device=device)
    return market_x, news_x


if __name__ == "__main__":
    # Quick smoke test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultimodalTransformer(n_market_features=6, news_emb_dim=768, d_model=128, n_heads=8, n_layers=2).to(device)
    market_x, news_x = make_dummy_input(batch=4, Tm=60, n_features=6, Tn=2, emb_dim=768, device=device)
    logits = model(market_x, news_x)
    print("Logits shape:", logits.shape)  # expect [B, 3]
