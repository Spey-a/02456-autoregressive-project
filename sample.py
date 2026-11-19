"""Generate protein sequences from a trained model.

    uv run python sample.py --checkpoint checkpoints/gru.pt --num 256
    uv run python sample.py --checkpoint checkpoints/gru.pt --num 256 --temperature 0.8

Writes results/sequences/<name>.csv with a `sequence` column -- the only thing eval/ asks
for. Score it with:

    uv run python eval/eval.py --samples results/sequences/<name>.csv
    bsub < hpc/eval.sh        # with --fold, on a GPU

HOW SAMPLING WORKS, and why it is the part training never exercises: during training the
model always sees the TRUE prefix (teacher forcing). Here it sees its own output, which is
a distribution it was never trained on. So a model with a good validation loss can still
generate poorly, and this is why you evaluate samples rather than trusting perplexity.

    start with <BOS>
    repeat:  logits = model.step(state, token)
             sample a token from softmax(logits / temperature)
             stop at <EOS>

Written against the step() interface, so it works with every model unchanged -- and in
weeks 3 and 4 the guided sampler is a few extra lines inside the same loop.
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from protgen.data import ProteinDataset
from protgen.models import build_model

PAD, BOS, EOS = 0, 1, 2


@torch.no_grad()
def generate(model, num, max_len=127, temperature=1.0, device="cpu", seed=0,
             batch_size=256):
    """Return a list of `num` amino-acid strings.

    Batched: the cost is in the max_len sequential steps, not in the batch dimension, so
    generating 256 at once is barely slower than generating one.
    """
    torch.manual_seed(seed)
    model.eval()
    out = []

    while len(out) < num:
        B = min(batch_size, num - len(out))
        state = None
        token = torch.full((B,), BOS, dtype=torch.long, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        rows = [[] for _ in range(B)]

        for _ in range(max_len):
            state, logits = model.step(state, token)
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).squeeze(1)
            # Finished rows keep emitting <EOS> so the batch stays rectangular.
            token = torch.where(done, torch.full_like(token, EOS), token)

            for b in range(B):
                t = int(token[b])
                if not done[b] and t not in (PAD, BOS, EOS):
                    rows[b].append(t)
            done |= token == EOS
            if bool(done.all()):
                break

        out.extend(rows)

    return out[:num]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num", type=int, default=256,
                   help="match the --num used for eval/make_controls.py")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="<1 sharpens (less diverse), >1 flattens. Hold at 1.0 while "
                        "sweeping guidance strength, or the two knobs confound.")
    p.add_argument("--out", default=None,
                   help="output CSV (default: results/sequences/<checkpoint name>.csv)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["model"], ckpt["vocab_size"],
                        max_len=ckpt["max_length"], **ckpt["kwargs"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded {args.checkpoint}  ({ckpt['model']}, {model.num_parameters:,} parameters)")

    seqs = generate(model, args.num, max_len=ckpt["max_length"] - 2,
                    temperature=args.temperature, device=device, seed=args.seed)

    # decode() lives on the dataset and drops the special tokens for us.
    decoder = ProteinDataset(f"{args.data_dir}/train.csv")
    strings = [decoder.decode(torch.tensor(s, dtype=torch.long)) for s in seqs]

    stem = Path(args.checkpoint).stem
    if args.temperature != 1.0:
        stem = f"{stem}_t{args.temperature}"
    out_path = Path(args.out) if args.out else Path("results/sequences") / f"{stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence"])
        w.writerows([[s] for s in strings])

    lengths = sorted(len(s) for s in strings)
    real = decoder.df.sequence.str.len()
    print(f"wrote {len(strings)} sequences to {out_path}")
    print(f"  generated lengths: min {lengths[0]}, median {lengths[len(lengths)//2]}, "
          f"max {lengths[-1]}")
    print(f"  real lengths     : min {real.min()}, median {int(real.median())}, "
          f"max {real.max()}")
    print("\nA generated length distribution far from the real one is itself a result:")
    print("the model has not learned when to emit <EOS>, which is a long-range property.")
    print(f"\nNext:  uv run python eval/eval.py --samples {out_path}")


if __name__ == "__main__":
    main()