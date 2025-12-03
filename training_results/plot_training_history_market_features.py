# plot_training_history_market_features.py

import os
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def assert_exists(p: Path, label: str):
    if not p.exists():
        parent = p.parent
        msg = [f"{label} not found: {p}"]
        if parent.exists():
            try:
                entries = "\n  - " + "\n  - ".join(sorted(e.name for e in parent.iterdir()))
                msg.append(f"\nContents of {parent}:\n{entries}")
            except Exception:
                pass
        else:
            msg.append(f"\nParent directory does not exist: {parent}")
        raise FileNotFoundError("\n".join(msg))


def load_history(history_path: Path):
    """
    Load a Keras-style history JSON:
    {
        "loss": [...],
        "val_loss": [...],  # optional
        ...
    }
    """
    assert_exists(history_path, "History JSON file")

    with history_path.open("r") as f:
        history = json.load(f)

    if "loss" not in history:
        raise KeyError(f"'loss' key not found in history JSON: {history_path}")

    loss = np.array(history["loss"], dtype=float)
    val_loss = None
    if "val_loss" in history:
        val_loss = np.array(history["val_loss"], dtype=float)

    return loss, val_loss, history


def plot_training_loss(loss, val_loss, out_path: Path, stock: str):
    """
    Plot training loss (and val_loss if present) vs epoch.
    """
    epochs = np.arange(1, len(loss) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, loss, label="loss", linewidth=1.8)

    if val_loss is not None:
        ax.plot(epochs, val_loss, label="val_loss", linewidth=1.8, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{stock} – Training loss over epochs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot training loss from <STOCK>_market_features_history.json and "
            "save to <base-dir>/training_results/<STOCK>/."
        )
    )
    ap.add_argument(
        "--stock", "--symbol", "-s",
        dest="stock",
        required=True,
        help="Stock symbol, e.g. TSLA, CSCO, INTC, PCLN.",
    )
    ap.add_argument(
        "--base-dir",
        default=r"C:\deep_hedging_VWAP",
        help="Project base directory (default: C:\\deep_hedging_VWAP).",
    )
    ap.add_argument(
        "--history-json",
        default="",
        help=(
            "Path to <STOCK>_market_features_history.json. "
            "Default: <base-dir>/data/<STOCK>/<STOCK>_market_features_history.json"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help=(
            "Directory to save the loss plot. "
            "Default: <base-dir>/training_results/<STOCK>."
        ),
    )
    args = ap.parse_args()

    stock = args.stock.strip().upper()
    base_dir = Path(args.base_dir)

    # Default history path: <base-dir>/data/<STOCK>/<STOCK>_market_features_history.json
    default_hist = base_dir / "data" / stock / f"{stock}_market_features_history.json"
    history_path = Path(args.history_json) if args.history_json else default_hist

    # Output dir: either user-provided or <base-dir>/training_results/<STOCK>
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = base_dir / "training_results" / stock

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Loading training history from: {history_path}")
    loss, val_loss, history = load_history(history_path)

    plot_path = out_dir / f"{stock}_training_loss.png"
    print(f"📉 Plotting training loss -> {plot_path}")
    plot_training_loss(loss, val_loss, plot_path, stock=stock)

    print("\n✅ Done. Training loss plot saved.")


if __name__ == "__main__":
    main()