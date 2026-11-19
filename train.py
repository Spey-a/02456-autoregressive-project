"""Train an autoregressive protein language model.

    uv run python train.py --model mlp   --epochs 10
    uv run python train.py --model gru   --epochs 15
    uv run python train.py --model transformer --epochs 15 --lr 1e-3

Writes a checkpoint to checkpoints/<name>.pt and a row of metrics to results/runs.csv,
so every run is recorded without anyone having to remember to write it down.

The loss is the categorical negative log-likelihood of the next token, averaged over
every position that has one. Positions past the end of a protein carry IGNORE_INDEX (-1)
in x_shifted and are skipped -- without that, the loss would be diluted by hundreds of
trivial padding predictions and the numbers would look good for no reason.

Reference points on this data (nats per residue, lower is better):
    2.996   uniform over the 20 residues
    2.803   amino-acid frequencies only
If a trained model does not beat 2.80, the bug is in the pipeline, not the architecture.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from protgen.data import IGNORE_INDEX, ProteinDataset
from protgen.models import build_model

UNIGRAM_BASELINE = 2.803          # nats/residue, fit on train, measured on test


def evaluate(model, dataset, device, batch_size=512):
    """Mean NLL per predicted token over a whole split, in nats.

    Summed over the split and divided once, rather than averaging batch averages:
    batches hold different numbers of real tokens, so averaging averages would weight
    short sequences too heavily.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            x = dataset.tokens[i:i + batch_size].to(device)
            target = dataset.tokens_shifted[i:i + batch_size].to(device)
            logits = model(x)
            total_loss += F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=IGNORE_INDEX, reduction="sum",
            ).item()
            total_tokens += (target != IGNORE_INDEX).sum().item()
    return total_loss / total_tokens


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mlp",
                   choices=["mlp", "gru", "lstm", "transformer"])
    p.add_argument("--name", default=None, help="checkpoint name (default: the model name)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=None, help="hidden size (MLP/RNN)")
    p.add_argument("--layers", type=int, default=None, help="layers (RNN/Transformer)")
    p.add_argument("--context", type=int, default=3, help="context width (MLP only)")
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip, 0 to disable")
    p.add_argument("--patience", type=int, default=3,
                   help="stop after this many epochs without validation improvement")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    name = args.name or args.model
    print(f"device: {device}")

    train_ds = ProteinDataset(f"{args.data_dir}/train.csv")
    val_ds = ProteinDataset(f"{args.data_dir}/val.csv")
    print(train_ds)

    # Only pass the options that apply to the chosen architecture.
    kwargs = {"dropout": args.dropout}
    if args.model == "mlp":
        kwargs["context"] = args.context
        if args.hidden:
            kwargs["hidden"] = args.hidden
    elif args.model in ("gru", "lstm"):
        if args.hidden:
            kwargs["hidden"] = args.hidden
        if args.layers:
            kwargs["layers"] = args.layers
    else:
        if args.hidden:
            kwargs["d_model"] = args.hidden
        if args.layers:
            kwargs["layers"] = args.layers

    model = build_model(args.model, train_ds.vocab_size,
                        max_len=train_ds.max_length, **kwargs).to(device)
    print(f"model: {args.model}  ({model.num_parameters:,} parameters)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path = Path("checkpoints") / f"{name}.pt"

    best_val, best_epoch, history = float("inf"), -1, []
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, target, _y, _length in loader:
            x, target = x.to(device), target.to(device)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            opt.zero_grad()
            loss.backward()
            if args.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            running += loss.item()
            batches += 1

        val = evaluate(model, val_ds, device)
        history.append({"epoch": epoch, "train": running / batches, "val": val})
        flag = ""
        if val < best_val:
            best_val, best_epoch = val, epoch
            torch.save({"model": args.model, "kwargs": kwargs, "seed": args.seed,
                        "vocab_size": train_ds.vocab_size,
                        "max_length": train_ds.max_length,
                        "state_dict": model.state_dict()}, ckpt_path)
            flag = "  *saved*"
        print(f"epoch {epoch:3d}  train {running/batches:.4f}  "
              f"val {val:.4f}  ppl {math.exp(val):6.2f}{flag}")

        if epoch - best_epoch >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping early")
            break

    elapsed = time.time() - start
    print(f"\nbest val {best_val:.4f} nats  (perplexity {math.exp(best_val):.2f}) "
          f"at epoch {best_epoch}, {elapsed:.0f}s")
    print(f"unigram baseline is {UNIGRAM_BASELINE:.3f} -- "
          f"{'BEATEN' if best_val < UNIGRAM_BASELINE else 'NOT BEATEN, check the pipeline'}")
    print(f"checkpoint: {ckpt_path}")

    # One row per run, so the results table writes itself.
    Path("results").mkdir(exist_ok=True)
    runs = Path("results/runs.csv")
    row = {"name": name, "model": args.model, "params": model.num_parameters,
           "epochs_run": len(history), "best_epoch": best_epoch,
           "val_nll": round(best_val, 4), "val_ppl": round(math.exp(best_val), 2),
           "lr": args.lr, "batch_size": args.batch_size, "dropout": args.dropout,
           "seed": args.seed, "seconds": round(elapsed), "kwargs": json.dumps(kwargs)}
    write_header = not runs.exists()
    with runs.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"appended a row to {runs}")


if __name__ == "__main__":
    main()