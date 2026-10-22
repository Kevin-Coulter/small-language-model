"""Plot train and val loss from a training log."""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="out/log.csv")
    p.add_argument("--out", default="out/loss.png")
    args = p.parse_args()

    train, val = [], []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            tokens = int(row["tokens"]) / 1e6
            if row["train_loss"]:
                train.append((tokens, float(row["train_loss"])))
            if row["val_loss"]:
                val.append((tokens, float(row["val_loss"])))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if train:
        ax.plot(*zip(*train), label="train")
    if val:
        ax.plot(*zip(*val), marker="o", markersize=3, label="val")
    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("Next-token prediction loss on WikiText-103")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
