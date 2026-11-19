"""Causal (decoder-only) Transformer language model -- a small protein GPT.

Like the RNN it can use the whole prefix, but it reads every earlier position directly
through attention instead of through a single compressed state. Nothing is squeezed into a
fixed-size vector, at the cost of attention being quadratic in sequence length (fine here:
T = 129).

Two details that are fatal if you get them wrong, and worth being able to explain:

  * THE CAUSAL MASK. generate_square_subsequent_mask(T) is 0 on and below the diagonal and
    -inf above it, added to the attention scores before the softmax, so position i gets
    exactly zero weight on every future position. Without it the model sees the answer:
    training loss collapses, validation perplexity looks wonderful, and the samples are
    garbage. `scripts/test_causal.py` catches this in three lines.

  * POSITIONAL ENCODINGS. Self-attention is permutation equivariant -- without position
    information a protein and a shuffle of it are the same input. Since amino-acid
    composition barely distinguishes the three families here, a model without positions
    would be close to useless.

No padding mask is needed. Padding sits only after <EOS>, and causal attention only ever
looks backwards, so a real position never attends to a padded one.
"""

import torch
import torch.nn as nn

from .base import ARModel


class TransformerLM(ARModel):
    def __init__(self, vocab_size, d_model=128, heads=4, layers=2,
                 max_len=129, ff_mult=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len

        self.emb = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, ff_mult * d_model,
            batch_first=True, norm_first=True, dropout=dropout,
        )
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, vocab_size)

    def _encode(self, x):
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.emb(x) + self.pos(positions)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.enc(h, mask=mask, is_causal=True)

    def forward(self, x):
        """(B, T) -> (B, T, V)."""
        return self.out(self._encode(x))

    def hidden_states(self, x):
        """(B, T) -> (B, T, d_model), for Phase 2's classifier head."""
        return self._encode(x)

    def step(self, state, token):
        """state is the prefix generated so far, as a (B, t) long tensor.

        Simple and correct: re-encode the prefix each step. That is O(t^2) per step rather
        than O(t) with a KV cache, but with T <= 129 and a 2-layer model it is not the
        bottleneck -- and it keeps the code short enough to trust. Add a cache later only
        if sampling actually becomes slow.
        """
        prefix = token[:, None] if state is None else torch.cat([state, token[:, None]], 1)
        prefix = prefix[:, -self.max_len:]           # never exceed the positional table
        logits = self.out(self._encode(prefix))[:, -1]   # (B, V): last position only
        return prefix, logits