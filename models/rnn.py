"""Discrete autoregressive RNN for next-token prediction over quantized bins."""

import torch.nn as nn


class DiscreteRNN(nn.Module):
    """
    tokens (B, T) -> Embedding -> RNN -> Linear -> logits (B, T, num_bins)

    Vanilla RNN baseline. The recurrence h_t = ReLU(W_xh e_t + W_hh h_{t-1})
    compresses all past context into a fixed hidden vector. Simple and fast,
    but struggles with long-range dependencies due to vanishing gradients.

    Args:
        num_bins:  size of the token vocabulary.
        d_model:   embedding dimension.
        hidden_dim: RNN hidden state size.
        n_layers:  number of stacked RNN layers.
    """

    def __init__(self, num_bins=128, d_model=64, hidden_dim=32, n_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, d_model)
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            nonlinearity="relu",
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, num_bins)

    def forward(self, x):
        # x: (B, T) long
        emb = self.embedding(x)        # (B, T, d_model)
        h, hidden = self.rnn(emb)      # (B, T, hidden_dim)
        logits = self.output(h)        # (B, T, num_bins)
        return logits, hidden
