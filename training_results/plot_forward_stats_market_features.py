# plot_forward_stats_market_features.py

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


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


def percentile_limits(W, alpha=0.001):
    """
    Return (xmin, xmax) so that we keep the central (1 - 2*alpha) fraction
    of the distribution, i.e. between the alpha and 1-alpha quantiles.
    """
    W = np.asarray(W).reshape(-1)
    try:
        xmin = np.quantile(W, alpha, method="linear")
        xmax = np.quantile(W, 1.0 - alpha, method="linear")
    except TypeError:
        xmin = np.quantile(W, alpha, interpolation="linear")
        xmax = np.quantile(W, 1.0 - alpha, interpolation="linear")

    if np.isclose(xmin, xmax):
        pad = 1e-6 if xmin == 0 else 1e-3 * abs(xmin)
        xmin, xmax = xmin - pad, xmax + pad

    return float(xmin), float(xmax)


def load_forward_stats(stats_h5: Path):
    """
    Load all datasets needed from <STOCK>_market_features_forward_stats.h5.
    """
    assert_exists(stats_h5, "Forward stats H5 file")

    with h5py.File(str(stats_h5), "r") as f:
        # Required datasets
        terminal_wealth = f["terminal_wealth"][:]          # (num_paths,)
        avg_delta_per_step = f["avg_delta_per_step"][:]    # (N,)
        terminal_delta = f["terminal_delta"][:]            # (num_paths,)
        avg_phi_per_step = f["avg_phi_per_step"][:]        # (N,)
        avg_theta_per_step = f["avg_theta_per_step"][:]    # (N, L_out)

        # Attributes
        stock = f.attrs.get("stock", None)
        N = int(f.attrs.get("N", avg_delta_per_step.shape[0]))
        n_policy_steps = int(f.attrs.get("n_policy_steps", avg_phi_per_step.shape[0]))
        L_out = int(f.attrs.get("L_out", avg_theta_per_step.shape[1]))

    meta = dict(
        stock=stock,
        N=N,
        n_policy_steps=n_policy_steps,
        L_out=L_out,
        num_paths=terminal_wealth.shape[0],
    )
    return terminal_wealth, avg_delta_per_step, terminal_delta, avg_phi_per_step, avg_theta_per_step, meta


def plot_terminal_wealth_kde(terminal_wealth, out_path: Path, title_prefix=""):
    """
    Smoothed terminal wealth distribution using Gaussian KDE and a central
    quantile window, e.g. central 99% of the distribution.
    """
    W = np.asarray(terminal_wealth).reshape(-1)

    kde_model = gaussian_kde(W)

    # Use central mass (e.g. 99.8%) by default via alpha
    xmin, xmax = percentile_limits(W, alpha=0.001)

    xx = np.linspace(xmin, xmax, 1000)
    kde_values = kde_model(xx)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xx, kde_values, linewidth=1.5, color="black", label="Wealth KDE")
    ax.fill_between(xx, kde_values, alpha=0.3, color="gray")

    ax.set_xlabel("Terminal wealth")
    ax.set_ylabel("Density")
    if title_prefix:
        ax.set_title(f"{title_prefix} Terminal wealth distribution (KDE)")
    else:
        ax.set_title("Terminal wealth distribution (KDE)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_avg_delta_trajectory(avg_delta_per_step, out_path: Path, title_prefix=""):
    """
    Average delta trajectory over N steps (simple line plot).
    """
    N = avg_delta_per_step.shape[0]
    steps = np.arange(N)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, avg_delta_per_step, marker="o", linewidth=1.5)
    ax.set_xlabel("Time step n")
    ax.set_ylabel("Average delta")
    if title_prefix:
        ax.set_title(f"{title_prefix} Average delta trajectory")
    else:
        ax.set_title("Average delta trajectory")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_terminal_delta_box(terminal_delta, out_path: Path, title_prefix=""):
    """
    Terminal delta distribution as a box-plot.
    """
    D = np.asarray(terminal_delta).reshape(-1)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(D, vert=True, showmeans=True)
    ax.set_xticks([1])
    ax.set_xticklabels(["Terminal Δ"])
    ax.set_ylabel("Terminal delta")

    if title_prefix:
        ax.set_title(f"{title_prefix} Terminal delta distribution")
    else:
        ax.set_title("Terminal delta distribution")

    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def plot_avg_policy_bars(avg_phi_per_step, avg_theta_per_step, out_path: Path, title_prefix=""):
    """
    Average policy trajectory (φ, θ₁, ..., θ_L) over N steps.
    Each policy component in its own subplot.
    Vertical bars per step (bar plot).
    """
    avg_phi_per_step = np.asarray(avg_phi_per_step)
    avg_theta_per_step = np.asarray(avg_theta_per_step)

    N = avg_phi_per_step.shape[0]
    L_out = avg_theta_per_step.shape[1]
    steps = np.arange(N)

    n_subplots = 1 + L_out  # phi + each theta level
    fig_height = max(2 * n_subplots, 6)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(10, fig_height), sharex=True)

    if n_subplots == 1:
        axes = [axes]

    # φ subplot
    ax0 = axes[0]
    ax0.bar(steps, avg_phi_per_step, width=0.8)
    ax0.set_ylabel("φ (MO)")
    ax0.set_title(f"{title_prefix} Average policy per step" if title_prefix else "Average policy per step")
    ax0.grid(True, axis="y", alpha=0.3)

    # θ subplots
    for k in range(L_out):
        ax = axes[k + 1]
        ax.bar(steps, avg_theta_per_step[:, k], width=0.8)
        ax.set_ylabel(f"θ{k+1}")
        ax.grid(True, axis="y", alpha=0.3)

    axes[-1].set_xlabel("Time step n")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Plot forward stats from <STOCK>_market_features_forward_stats.h5:\n"
            " 1) Smoothed terminal wealth distribution (KDE)\n"
            " 2) Average delta trajectory\n"
            " 3) Terminal delta box-plot\n"
            " 4) Average policy trajectories (φ, θ₁..θ_L) as bar plots per step"
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
        "--stats-h5",
        default="",
        help=(
            "Path to <STOCK>_market_features_forward_stats.h5. "
            "Default: <base-dir>/data/<STOCK>/<STOCK>_market_features_forward_stats.h5"
        ),
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help=(
            "Directory to save plots. "
            "Default: <base-dir>/training_results/<STOCK>."
        ),
    )
    args = ap.parse_args()

    stock = args.stock.strip().upper()
    base_dir = Path(args.base_dir)

    # Default stats path: <base-dir>/data/<STOCK>/<STOCK>_market_features_forward_stats.h5
    default_stats = base_dir / "data" / stock / f"{stock}_market_features_forward_stats.h5"
    stats_h5 = Path(args.stats_h5) if args.stats_h5 else default_stats

    # Output dir: either user-provided, or <base-dir>/training_results/<STOCK>
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = base_dir / "training_results" / stock

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 Loading forward stats from: {stats_h5}")
    (terminal_wealth,
     avg_delta_per_step,
     terminal_delta,
     avg_phi_per_step,
     avg_theta_per_step,
     meta) = load_forward_stats(stats_h5)

    stock_attr = meta["stock"] if meta["stock"] else stock
    title_prefix = stock_attr

    # 1) Terminal wealth distribution (smoothed KDE)
    wealth_plot = out_dir / f"{stock_attr}_wealth_kde.png"
    print(f"📊 Plotting terminal wealth KDE -> {wealth_plot}")
    plot_terminal_wealth_kde(terminal_wealth, wealth_plot, title_prefix=title_prefix)

    # 2) Average delta trajector
    delta_traj_plot = out_dir / f"{stock_attr}_avg_delta_trajectory.png"
    print(f"📈 Plotting average delta trajectory -> {delta_traj_plot}")
    plot_avg_delta_trajectory(avg_delta_per_step, delta_traj_plot, title_prefix=title_prefix)

    # 3) Terminal delta distribution (box-plot)
    delta_box_plot = out_dir / f"{stock_attr}_terminal_delta_boxplot.png"
    print(f"📦 Plotting terminal delta box-plot -> {delta_box_plot}")
    plot_terminal_delta_box(terminal_delta, delta_box_plot, title_prefix=title_prefix)

    # 4) Average policy trajectories (bars per step)
    policy_plot = out_dir / f"{stock_attr}_avg_policy_bars.png"
    print(f"📚 Plotting average policy trajectories -> {policy_plot}")
    plot_avg_policy_bars(avg_phi_per_step, avg_theta_per_step, policy_plot, title_prefix=title_prefix)

    print("\n✅ Done. All plots saved.")


if __name__ == "__main__":
    main()
