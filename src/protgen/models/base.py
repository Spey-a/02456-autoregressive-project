"""The interface every autoregressive model in this project implements.

Two methods, and the whole project depends on them meaning the same thing for every
architecture:

    forward(x)            -> (B, T, V) logits.  Used in TRAINING, where the true prefix is
                             known, so all T positions are computed in one pass
                             ("teacher forcing").

    step(state, token)    -> (new_state, (B, V) logits).  Used in SAMPLING, where the prefix
                             is whatever the model just produced, so steps are sequential.
                             `state` is whatever that architecture needs to carry forward;
                             `None` means "start of sequence".

Because the signatures match, `sample.py` is written once and works for all three models --
and in weeks 3 and 4 the guided sampler will be too.

Position i of the output predicts x_shifted[i], i.e. the token at position i+1 of the input.
"""

import torch.nn as nn


class ARModel(nn.Module):
    """Base class. Subclasses implement forward() and step()."""

    def forward(self, x):
        """x: (B, T) long -> (B, T, V) logits."""
        raise NotImplementedError

    def step(self, state, token):
        """One sampling step.

        state: architecture-specific, or None at the start of a sequence.
        token: (B,) long -- the token just produced (or <BOS> on the first call).
        returns (new_state, logits) with logits (B, V) for the NEXT token.
        """
        raise NotImplementedError

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())