"""Autoregressive models for the protein guidance project.

Every model implements the same two methods -- see base.py:

    forward(x)         -> (B, T, V)   for training
    step(state, token) -> (state, (B, V))  for sampling

so train.py and sample.py work with any of them unchanged.
"""

from .base import ARModel
from .mlp import MLPLM
from .rnn import RNNLM
from .transformer import TransformerLM

__all__ = ["ARModel", "MLPLM", "RNNLM", "TransformerLM", "build_model"]


def build_model(name, vocab_size, max_len=129, **kwargs):
    """Make a model by name, so train.py can take --model from the command line."""
    name = name.lower()
    if name == "mlp":
        return MLPLM(vocab_size, **kwargs)
    if name in ("rnn", "gru"):
        return RNNLM(vocab_size, cell="gru", **kwargs)
    if name == "lstm":
        return RNNLM(vocab_size, cell="lstm", **kwargs)
    if name in ("transformer", "gpt"):
        return TransformerLM(vocab_size, max_len=max_len, **kwargs)
    raise ValueError(f"unknown model {name!r}; expected mlp, gru, lstm or transformer")