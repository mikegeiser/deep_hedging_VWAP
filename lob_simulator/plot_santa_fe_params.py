# plot_santa_fe_params.py

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Adjust import path if needed depending on your project layout.
# Here we assume santa_fe_param.py is importable directly.
from santa_fe_param import _load_santa_fe_csv


def plot_intensities(lambda_vec, rho_vec, a_0_vec, K, out_path, stock):
    """
    Plot lambda(k) and rho(k) * ask_queue(k) vs relative price k = 1..K.

    Legend labels: "lambda" and "rho" (even though rho is scaled by ask_queue).
    """
    k = np.arange(1, K + 1)
    lam = lambda_vec[:K]
    rho_eff = rho_vec[:K] * a_0_vec[:K]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k, lam, label="lambda", linewidth=1.8)
    ax.plot(k, rho_eff, label="rho", linewidth=1.8)

    ax.set_xlabel("Relative price level k")
    ax.set_ylabel("Intensity")
    ax.set_title(f"{stock} – Santa Fe intensities")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_queues(a_0_vec, b_0_vec, K, out_path, stock):
    """
    Plot initial ask and bid queues vs relative price, using vertical bars.

    For bids, x = -k; for asks, x = +k, k = 1..K.
    """
    k = np.arange(1, K + 1)
    x_bid = -k
    x_ask = k

    ask_q = a_0_vec[:K]
    bid_q = b_0_vec[:K]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Vertical bars: bids at negative k, asks at positive k
    ax.bar(x_ask, ask_q, width=0.8, label="ask queue")
    ax.bar(x_bid, bid_q, width=0.8, label="bid queue")

    ax.set_xlabel("Relative price (ticks)")
    ax.set_ylabel("Queue size")
    ax.set_title(f"{stock} – Initial ask/bid queues")
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot Santa Fe parameters for a given stock:\n"
            " 1) Intensities: lambda(k) and rho(k)*ask_queue(k) vs relative price k\n"
            " 2) Queues: ask and bid queues vs relative price (bid at -k, ask at +k) with bars"
        )
    )
    ap.add_argument(
        "--stock", "--symbol", "-s",
        dest="stock",
        default=os.getenv("LOB_STOCK", "TSLA"),
        help="Stock symbol, e.g. TSLA, CSCO, INTC, PCLN. "
             "Default: env LOB_STOCK or TSLA.",
    )
    ap.add_argument(
        "--base-dir",
        default=r"C:\deep_hedging_VWAP",
        help="Project base directory (used only for output paths; "
             "Santa Fe CSV path is resolved by santa_fe_param.py).",
    )
    args = ap.parse_args()

    stock = args.stock.strip().upper()
    base_dir = Path(args.base_dir)

    # Load Santa Fe parameters via your existing loader
    params = _load_santa_fe_csv(stock)
    K = int(params["K"])
    K_trunc = int(params["K_trunc"])  # still loaded, but we now use full K
    lambda_vec = params["lambda_vec"].astype(float)
    rho_vec = params["rho_vec"].astype(float)
    a_0_vec = params["a_0_vec"].astype(float)
    b_0_vec = params["b_0_vec"].astype(float)

    print(f"[~] Loaded Santa Fe params for {stock}: K={K}, K_trunc={K_trunc}")

    # Where to save plots: <base-dir>/data/<STOCK>/
    out_dir = base_dir / "data" / stock
    out_dir.mkdir(parents=True, exist_ok=True)

    intensities_png = out_dir / f"{stock}_santa_fe_intensities.png"
    queues_png = out_dir / f"{stock}_santa_fe_queues.png"

    print(f"📊 Plotting intensities (using full K={K}) -> {intensities_png}")
    plot_intensities(lambda_vec, rho_vec, a_0_vec, K, intensities_png, stock)

    print(f"📚 Plotting queues (using full K={K}) -> {queues_png}")
    plot_queues(a_0_vec, b_0_vec, K, queues_png, stock)

    print("✅ Done. Plots saved.")


if __name__ == "__main__":
    main()