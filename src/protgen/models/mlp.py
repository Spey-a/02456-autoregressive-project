"""Fixed-context MLP language model (Bengio et al., 2003).

Approximates  p(x_i | x_<i)  by  p(x_i | x_{i-m}, ..., x_{i-1}).

It is a neural n-gram model: instead of a count table over contexts, each token gets a
learned vector, the m context vectors are concatenated, and an MLP maps that to a
distribution over the next token. The win over counting is that similar residues get
similar vectors, so the model generalises to contexts it never saw.

Its limitation is the point of including it: it cannot represent any dependence on
x_{i-m-1} or earlier. Proteins fold, so residues far apart in sequence are often close in
space and strongly correlated. Compare its validation loss with the GRU's to see what
that costs.
"""

import torch
import torch.nn as nn

from .base import ARModel


class MLPLM(ARModel):
    def __init__(self, vocab_size, context=3, emb_dim=32, hidden=256, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.context = context

        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(context * emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, x):
        """(B, T) -> (B, T, V)."""
        B, T = x.shape
        # Left-pad with <PAD>=0 so position 0 has a full context window.
        pad = x.new_zeros(B, self.context - 1)
        padded = torch.cat([pad, x], dim=1)              # (B, T + context - 1)
        # unfold builds every sliding window of width `context` as a view, so all T
        # positions are computed in one batched pass instead of a Python loop.
        windows = padded.unfold(1, self.context, 1)      # (B, T, context)
        e = self.emb(windows).flatten(-2)                # (B, T, context * emb_dim)
        return self.net(e)

    def step(self, state, token):
        """state is the last (context-1) tokens, as a (B, context-1) long tensor."""
        B = token.shape[0]
        if state is None:
            # Nothing generated yet: pad the window with <PAD>=0.
            state = token.new_zeros(B, self.context - 1)
        window = torch.cat([state, token[:, None]], dim=1)   # (B, context)
        e = self.emb(window).flatten(-2)                     # (B, context * emb_dim)
        logits = self.net(e)                                 # (B, V)
        return window[:, 1:], logits                         # drop the oldest token