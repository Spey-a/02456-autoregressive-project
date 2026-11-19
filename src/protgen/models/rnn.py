"""Recurrent language model (GRU or LSTM).

The hidden state h_i is a learned, fixed-size summary of the ENTIRE prefix x_<=i, so the
context is unbounded in principle -- bounded in practice only by the capacity of that
vector. This is the main qualitative difference from the MLP, and it is where most of the
improvement in validation loss comes from.

Why GRU rather than a plain RNN: backpropagation through time multiplies by the recurrent
weight matrix at every step, so gradients vanish or explode over long sequences. The GRU's
gated, additive state update gives gradients a path that does not get repeatedly rescaled
-- the same trick as a residual connection.

The trunk also does double duty. In Phase 2 the classifier head reads exactly these hidden
states: h_i depends on x_<=i, so it is already a representation of the i-th prefix, which
is what a classifier guiding generation needs.
"""

import torch.nn as nn

from .base import ARModel


class RNNLM(ARModel):
    def __init__(self, vocab_size, emb_dim=32, hidden=256, layers=1,
                 cell="gru", dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.layers = layers
        self.cell = cell.lower()

        self.emb = nn.Embedding(vocab_size, emb_dim)
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[self.cell]
        self.rnn = rnn_cls(
            emb_dim, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        """(B, T) -> (B, T, V). One call computes all T steps (teacher forcing)."""
        h, _ = self.rnn(self.emb(x))
        return self.out(self.drop(h))

    def hidden_states(self, x):
        """(B, T) -> (B, T, hidden).

        The per-prefix representations, without the output projection. Phase 2's
        classifier head reads these.
        """
        h, _ = self.rnn(self.emb(x))
        return h

    def step(self, state, token):
        """state is the RNN's hidden state (a tensor for GRU, a tuple for LSTM)."""
        h, new_state = self.rnn(self.emb(token).unsqueeze(1), state)
        return new_state, self.out(h.squeeze(1))