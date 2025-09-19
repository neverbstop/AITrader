import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism from the Transformer paper.
    This allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension of the query, key, and value vectors
        
        # Linear layers for Queries, Keys, and Values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Linear layer for the final output
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Calculates the scaled dot-product attention.
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Mask out padded positions
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def split_heads(self, x):
        """
        Splits the input tensor into multiple heads for multi-head attention.
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        Combines the multiple heads back into a single tensor.
        """
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. Linear projections to get Q, K, V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 2. Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # 3. Apply scaled dot-product attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. Combine heads and pass through a final linear layer
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    """
    A simple two-layer feed-forward network with a ReLU activation in between.
    """
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the embeddings, as transformers do not
    inherently understand word order.
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    """
    Represents a single encoder layer of the transformer, combining multi-head 
    attention and a feed-forward network with residual connections and layer normalization.
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        # Multi-Head Attention block
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Position-wise Feed-Forward block
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class SentimentTransformer(nn.Module):
    """
    The complete sentiment analysis model for text classification.
    It combines an embedding layer, positional encoding, a stack of
    TransformerEncoderLayers, and a final classification head.
    """
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout, num_classes):
        super(SentimentTransformer, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.d_model)  # Scale embeddings
        src = self.pos_encoder(src)
        
        # Pass through the encoder layers
        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_mask)

        # The final output is usually from the first token's (e.g., [CLS] token) representation
        # which is at index 0.
        output = output[:, 0, :]

        return self.fc(self.dropout(output))
