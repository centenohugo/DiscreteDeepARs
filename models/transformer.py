"""Discrete autoregressive LSTM-Transformer hybrid for next-token prediction."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding (Vaswani et al., 2017).

    Attention is permutation-invariant, so without positional information the
    Transformer cannot distinguish token order. These sinusoids with varying
    frequency let the model recover position from the embedding itself.
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Buffer: moves with .to(device) but is not a learnable parameter
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        return x + self.pe[:, : x.size(1)]


class DiscreteLSTMTransformer(nn.Module):
    """
    Hybrid model that mirrors the Gaussian baseline architecture:

        tokens -> Embedding -> LSTM -> Linear(proj) -> +PE
               -> TransformerEncoder (causal) -> Linear -> logits

    The LSTM cheaply extracts local temporal features; the Transformer then
    models global, long-range interactions between those features. A causal
    mask in the attention prevents the model from peeking at future tokens,
    which is what makes the next-token objective non-trivial.

    Args:
        num_bins:        size of the token vocabulary.
        d_model:         Transformer model dimension (also the embedding dim).
        nhead:           number of attention heads.
        num_layers:      number of Transformer encoder layers.
        dim_feedforward: hidden size of the feedforward sub-block.
        hidden_size:     LSTM hidden size.
        lstm_layers:     number of LSTM layers.
        max_seq_len:     maximum sequence length supported by PositionalEncoding.
        dropout:         dropout used inside the Transformer encoder.
    """

    def __init__(self,
                 num_bins=128,
                 d_model=64,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=16,
                 hidden_size=32,
                 lstm_layers=1,
                 max_seq_len=300,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Bring LSTM output back up to d_model so the Transformer can consume it
        self.proj = nn.Linear(hidden_size, d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output = nn.Linear(d_model, num_bins)

    def forward(self, x):
        # x: (B, T) long
        B, T = x.shape

        emb = self.embedding(x)                   # (B, T, d_model)
        h_lstm, _ = self.lstm(emb)                # (B, T, hidden_size)
        h = self.proj(h_lstm)                     # (B, T, d_model)
        h = self.pos_encoding(h)                  # add positional info

        # Causal mask: at step t, attend only to steps <= t
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(h, mask=causal_mask, is_causal=True)   # (B, T, d_model)

        logits = self.output(h)                   # (B, T, num_bins)
        return logits, None
