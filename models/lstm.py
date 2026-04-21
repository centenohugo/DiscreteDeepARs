"""Discrete autoregressive LSTM for next-token prediction over quantized bins."""

import torch.nn as nn


class DiscreteLSTM(nn.Module):
    """
    tokens (B, T) -> Embedding -> LSTM -> Linear -> logits (B, T, num_bins)

    The LSTM cell maintains an extra cell state c_t that is updated additively
    through learned gates (forget, input, output). Gradients flow along c_t
    without being repeatedly crushed by tanh activations, which makes the LSTM
    much more capable of learning long-range dependencies than a vanilla RNN.

    Args:
        num_bins:   size of the token vocabulary.
        d_model:    embedding dimension.
        hidden_dim: LSTM hidden state size.
        n_layers:   number of stacked LSTM layers.
        drop_prob:  dropout between stacked layers (only applied if n_layers > 1).
    """

    def __init__(self, num_bins=128, d_model=64, hidden_dim=32, n_layers=1, drop_prob=0.3):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=drop_prob if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        # x: (B, T) long
        emb = self.embedding(x)        # (B, T, d_model)
        h, hidden = self.lstm(emb)     # h: (B, T, hidden_dim); hidden: (h_n, c_n)
        logits = self.output(h)        # (B, T, num_bins)
        return logits, hidden
