"""Check that every model is causal, and that forward() and step() agree.

    uv run python scripts/test_causal.py

Run this whenever anyone touches a model. Both bugs it catches are invisible in the loss
curve:

  * A LEAKED FUTURE. If output i depends on x_{>i}, the model is being shown the answer.
    Training loss collapses, validation perplexity looks excellent, and the samples are
    garbage. The usual cause is a forgotten causal mask in the Transformer.

  * forward/step DISAGREEMENT. You train with forward() and sample with step(). If they
    compute different things, you evaluate a model you never trained, and nothing in the
    training logs will tell you.
"""

import sys

import torch

from protgen.data import ProteinDataset
from protgen.models import build_model

MODELS = ["mlp", "gru", "lstm", "transformer"]


def main():
    ds = ProteinDataset("data/train.csv")
    V, T = ds.vocab_size, ds.max_length
    x = ds.tokens[:4]
    failures = []

    for name in MODELS:
        torch.manual_seed(0)
        model = build_model(name, V, max_len=T)
        model.eval()

        with torch.no_grad():
            # --- causality: perturbing position p must not move outputs before p ---
            p = 50
            a = model(x)
            x2 = x.clone()
            x2[:, p] = (x2[:, p] + 1) % V
            b = model(x2)
            moved = ((a - b).abs().sum(-1)[0] > 1e-6).nonzero().flatten()
            first = int(moved.min()) if moved.numel() else T
            causal_ok = first >= p

            # --- forward/step agreement on the first few positions ---
            state, step_logits = None, []
            for i in range(12):
                state, logits = model.step(state, x[:, i])
                step_logits.append(logits)
            step_logits = torch.stack(step_logits, dim=1)       # (B, 12, V)
            max_diff = (a[:, :12] - step_logits).abs().max().item()
            agree_ok = max_diff < 1e-4

        status = "ok" if (causal_ok and agree_ok) else "FAIL"
        print(f"{name:12s} {status:5s}  first moved output {first:3d} (want >= {p})"
              f"   max |forward - step| = {max_diff:.2e}")
        if not causal_ok:
            failures.append(f"{name}: output {first} depends on position {p} (future leak)")
        if not agree_ok:
            failures.append(f"{name}: forward and step disagree by {max_diff:.2e}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        sys.exit(1)
    print("all models are causal and forward/step agree")


if __name__ == "__main__":
    main()