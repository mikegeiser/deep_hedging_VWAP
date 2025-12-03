# extract_market_features.py

import os
import argparse
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

from model import model_hedge_strat, policy_probe  # adjust import if needed


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


def load_snapshots(mkt_h5: Path):
    """
    Load 'snapshots' from <STOCK>_market_features.h5 and basic attrs.

    Returns
    -------
    snaps : np.ndarray, shape (num_paths, N+1, D)
    meta  : dict with keys N, D, L, num_paths, num_snaps
    """
    with h5py.File(str(mkt_h5), "r") as f:
        if "snapshots" not in f:
            raise KeyError(f"'snapshots' dataset not found in {mkt_h5}. Available: {list(f.keys())}")
        snaps = f["snapshots"][:]  # (num_paths, N+1, D)
        num_paths, num_snaps, D = snaps.shape

        N_attr = int(f.attrs.get("N", num_snaps - 1))
        D_attr = int(f.attrs.get("D", D))
        L = int(f.attrs.get("L", 0))

        if D_attr != D:
            raise ValueError(f"D mismatch in attrs vs data: D_attr={D_attr}, D_data={D}")
        if num_snaps != N_attr + 1:
            raise ValueError(f"Time dimension mismatch: num_snaps={num_snaps}, but attrs say N={N_attr} (expect N+1).")
        if L <= 0:
            raise ValueError(f"Bad L attribute in {mkt_h5}: L={L}")

    meta = dict(N=N_attr, D=D, L=L, num_paths=num_paths, num_snaps=num_snaps)
    return snaps, meta


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run a single forward pass of the trained model on ALL training data "
            "and export summary stats to a single HDF5 file, using *_market_features.*:\n"
            "  - terminal wealth per trajectory\n"
            "  - average delta per time step\n"
            "  - terminal delta per trajectory\n"
            "  - average phi per time step\n"
            "  - average theta per level per time step"
        )
    )
    # --stock / --symbol / -s all map to the same thing
    ap.add_argument(
        "--stock", "--symbol", "-s",
        dest="stock",
        required=True,
        help="Stock symbol, e.g. TSLA, CSCO, INTC, PCLN.",
    )
    ap.add_argument(
        "--market-h5",
        default="",
        help=(
            "Path to <STOCK>_market_features.h5. "
            "Default: <project_root>/data/<STOCK>/<STOCK>_market_features.h5"
        ),
    )
    ap.add_argument(
        "--weights",
        default="",
        help=(
            "Path to weights .h5. "
            "Default: <project_root>/data/<STOCK>/<STOCK>_market_features_best.weights.h5"
        ),
    )
    ap.add_argument(
        "--out-h5",
        default="",
        help=(
            "Output HDF5 path. "
            "Default: <project_root>/data/<STOCK>/<STOCK>_market_features_forward_stats.h5"
        ),
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size for model.predict (default: 1024).",
    )
    args = ap.parse_args()

    stock = args.stock.strip().upper()

    # project_root = parent of the folder containing this script
    # if script is C:\deep_hedging_VWAP\deep_hedging\..., then project_root = C:\deep_hedging_VWAP
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / stock

    # Default paths under <project_root>/data/<STOCK>/
    market_h5 = Path(args.market_h5) if args.market_h5 else (
        data_dir / f"{stock}_market_features.h5"
    )
    weights_path = Path(args.weights) if args.weights else (
        data_dir / f"{stock}_market_features_best.weights.h5"
    )
    out_h5 = Path(args.out_h5) if args.out_h5 else (
        data_dir / f"{stock}_market_features_forward_stats.h5"
    )
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Project root auto-detected as: {project_root}")
    print(f"📂 Data directory for {stock}: {data_dir}")

    print("\n🔁 Checking input paths...")
    assert_exists(market_h5, "Market features H5 file")
    assert_exists(weights_path, "Weights file")

    print(f"\n📦 Loading snapshots from: {market_h5}")
    snaps, meta = load_snapshots(market_h5)
    num_paths = meta["num_paths"]
    num_snaps = meta["num_snaps"]  # should be N+1
    D_file = meta["D"]
    N_attr = meta["N"]
    L_attr = meta["L"]

    print(f"  snapshots shape: {snaps.shape} (paths, snaps, D)")
    print(f"  N={N_attr}, D={D_file}, L={L_attr}")

    # Check model vs data feature width
    F_model = int(model_hedge_strat.inputs[0].shape[-1])
    if F_model != D_file:
        raise ValueError(f"Model expects feature dim={F_model} but data has D={D_file}.")

    print(f"\n🔁 Loading weights from: {weights_path}")
    model_hedge_strat.load_weights(str(weights_path))

    # Build model inputs for all time steps: list length = num_snaps (N+1)
    print("\n🧱 Building x_eval (full dataset for all time steps)...")
    x_eval = [snaps[:, t, :].astype(np.float32) for t in range(num_snaps)]

    # Collect delta_acc_* outputs from the main model
    print("\n🔍 Collecting delta_acc_* outputs from model...")
    delta_outputs = []
    n = 0
    while True:
        layer_name = f"delta_acc_{n}"
        try:
            layer = model_hedge_strat.get_layer(layer_name)
        except ValueError:
            break
        delta_outputs.append(layer.output)
        n += 1

    if not delta_outputs:
        raise RuntimeError("No 'delta_acc_*' layers found in model_hedge_strat.")

    num_delta_steps = len(delta_outputs)
    print(f"  Found {num_delta_steps} delta_acc_* layers.")

    # policy_probe gives [phi_0,...,phi_{N-1}, theta_0,...,theta_{N-1}]
    print("\n🧠 Preparing combined probe model (wealth + deltas + policy outputs) ...")
    combined_outputs = [model_hedge_strat.output] + delta_outputs + list(policy_probe.outputs)

    combined_probe = tf.keras.Model(
        inputs=model_hedge_strat.inputs,
        outputs=combined_outputs,
        name="forward_stats_probe"
    )

    # Single forward pass
    print("\n🚀 Running single forward pass over ALL trajectories...")
    all_outs = combined_probe.predict(x_eval, batch_size=args.batch_size, verbose=1)

    # Unpack outputs
    wealth_out = all_outs[0]  # terminal wealth, shape (num_paths, 1) or (num_paths,)
    wealth_out = np.asarray(wealth_out).reshape(num_paths)

    delta_list = all_outs[1:1 + num_delta_steps]  # each (num_paths, 1)
    policy_outs = all_outs[1 + num_delta_steps:]  # [phi_0..phi_{N-1}, theta_0..theta_{N-1}]

    # Deltas: stack into (num_paths, num_delta_steps)
    delta_mat = np.column_stack([np.asarray(d).reshape(num_paths) for d in delta_list])

    # 1) Terminal wealth per trajectory
    terminal_wealth = wealth_out  # shape (num_paths,)

    # 2) Average delta per time step over trajectories
    avg_delta_per_step = delta_mat.mean(axis=0)  # shape (num_delta_steps,)

    # 3) Terminal delta per trajectory
    terminal_delta = delta_mat[:, -1]  # shape (num_paths,)

    # 4) Average allocations for φ (MOs) and θ (LO vector) per time step
    num_policy_outs = len(policy_outs)
    # With snapshots of length N+1, we expect N policy steps for phi/theta
    n_phi = num_snaps - 1
    if num_policy_outs < 2 * n_phi:
        raise RuntimeError(
            f"Expected at least {2 * n_phi} policy outputs (phi+theta) but got {num_policy_outs}."
        )

    phi_outs = policy_outs[:n_phi]
    theta_outs = policy_outs[n_phi: n_phi + n_phi]

    # Infer L_out from a theta sample
    theta_sample = np.asarray(theta_outs[0])
    if theta_sample.ndim != 2:
        raise RuntimeError(f"Unexpected theta output shape: {theta_sample.shape}")
    L_out = theta_sample.shape[1]

    # avg_phi_per_step: (n_phi,)
    avg_phi_per_step = np.array(
        [np.asarray(phi_t).reshape(num_paths).mean() for phi_t in phi_outs],
        dtype=np.float64,
    )

    # avg_theta_per_step: (n_phi, L_out)
    avg_theta_per_step = np.zeros((n_phi, L_out), dtype=np.float64)
    for t, theta_t in enumerate(theta_outs):
        arr = np.asarray(theta_t)  # (num_paths, L_out)
        avg_theta_per_step[t, :] = arr.mean(axis=0)

    # ---------- Save everything to a single HDF5 file ----------
    print(f"\n💾 Saving forward stats to: {out_h5}")
    with h5py.File(str(out_h5), "w") as f:
        # Attributes for context
        f.attrs["stock"] = stock
        f.attrs["num_paths"] = num_paths
        f.attrs["num_snaps"] = num_snaps
        f.attrs["D"] = D_file
        f.attrs["N"] = N_attr
        f.attrs["L_market_features"] = L_attr
        f.attrs["num_delta_steps"] = num_delta_steps
        f.attrs["n_phi_steps"] = n_phi
        f.attrs["L_out"] = L_out

        # 1) terminal wealth per trajectory
        f.create_dataset("terminal_wealth", data=terminal_wealth, compression="lzf")

        # 2) average delta per time step
        f.create_dataset("avg_delta_per_step", data=avg_delta_per_step, compression="lzf")

        # 3) terminal delta per trajectory
        f.create_dataset("terminal_delta", data=terminal_delta, compression="lzf")

        # 4a) average φ per time step (market order allocation)
        f.create_dataset("avg_phi_per_step", data=avg_phi_per_step, compression="lzf")

        # 4b) average θ per level per time step
        for k in range(L_out):
            ds_name = f"avg_theta_level_{k + 1}_per_step"
            f.create_dataset(ds_name, data=avg_theta_per_step[:, k], compression="lzf")

        # Optional: full matrix
        f.create_dataset("avg_theta_per_step", data=avg_theta_per_step, compression="lzf")

    print("✅ Done. All requested stats stored in a single HDF5 file.")


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    tf.get_logger().setLevel("ERROR")
    main()