# export_forward_stats_market_features.py

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
        L_attr = int(f.attrs.get("L", 0))

        if D_attr != D:
            raise ValueError(f"D mismatch in attrs vs data: D_attr={D_attr}, D_data={D}")
        if num_snaps != N_attr + 1:
            raise ValueError(
                f"Time dimension mismatch: num_snaps={num_snaps}, but attrs say N={N_attr} (expect N+1)."
            )

        # If L attribute is missing or 0, infer it from D using D = 3*L + 4
        if L_attr <= 0:
            if (D - 4) % 3 != 0:
                raise ValueError(
                    f"Cannot infer L from D={D} in {mkt_h5}: (D - 4) is not divisible by 3."
                )
            L = (D - 4) // 3
            print(f"⚠️  L attribute is {L_attr} in {mkt_h5}; inferring L={L} from D={D}.")
        else:
            L = L_attr

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
        "--base-dir",
        default=r"C:\deep_hedging_VWAP",
        help="Project base directory (default: C:\\deep_hedging_VWAP).",
    )
    ap.add_argument(
        "--weights",
        default="",
        help=(
            "Path to weights .h5 "
            "(default: <base-dir>/data/<STOCK>/<STOCK>_market_features_best.weights.h5)."
        ),
    )
    ap.add_argument(
        "--market-h5",
        default="",
        help=(
            "Path to <STOCK>_market_features.h5 "
            "(default: <base-dir>/data/<STOCK>/<STOCK>_market_features.h5)."
        ),
    )
    ap.add_argument(
        "--out-h5",
        default="",
        help=(
            "Output HDF5 path "
            "(default: <base-dir>/data/<STOCK>/<STOCK>_market_features_forward_stats.h5)."
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
    base_dir = Path(args.base_dir)

    data_dir = base_dir / "data" / stock

    # Default paths based on *_market_features under data/
    weights_path = Path(args.weights) if args.weights else (
        data_dir / f"{stock}_market_features_best.weights.h5"
    )
    market_h5 = Path(args.market_h5) if args.market_h5 else (
        data_dir / f"{stock}_market_features.h5"
    )
    out_h5 = Path(args.out_h5) if args.out_h5 else (
        data_dir / f"{stock}_market_features_forward_stats.h5"
    )
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    print("\n🔁 Checking input paths...")
    assert_exists(weights_path, "Weights file")
    assert_exists(market_h5, "Market features H5 file")

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

    # policy_probe gives policy-related outputs; we don't assume their count,
    # we classify them by shape into phi (scalar) and theta (L-dimensional).
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
    policy_outs = all_outs[1 + num_delta_steps:]  # policy-related outputs

    # Deltas: stack into (num_paths, num_delta_steps)
    delta_mat = np.column_stack([np.asarray(d).reshape(num_paths) for d in delta_list])

    # 1) Terminal wealth per trajectory
    terminal_wealth = wealth_out  # shape (num_paths,)

    # 2) Average delta per time step over trajectories
    avg_delta_per_step = delta_mat.mean(axis=0)  # shape (num_delta_steps,)

    # 3) Terminal delta per trajectory
    terminal_delta = delta_mat[:, -1]  # shape (num_paths,)

    # 4) Classify policy outputs as φ (scalar) or θ (vector) based on shape
    phi_outs = []
    theta_outs = []

    for i, out in enumerate(policy_outs):
        arr = np.asarray(out)
        if arr.ndim == 1:
            # (num_paths,) -> treat as (num_paths, 1)
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise RuntimeError(
                f"Unexpected policy output shape at index {i}: {arr.shape} "
                "(expected 2D: (num_paths, d))."
            )

        d = arr.shape[1]
        if d == 1:
            phi_outs.append(arr)      # φ: scalar per path
        else:
            theta_outs.append(arr)    # θ: L-dimensional per path

    if not phi_outs:
        raise RuntimeError("No scalar policy outputs (phi) detected in policy_probe.outputs.")
    if not theta_outs:
        raise RuntimeError("No vector policy outputs (theta) detected in policy_probe.outputs.")

    n_phi = len(phi_outs)
    n_theta = len(theta_outs)
    print(f"\n📊 Detected {n_phi} phi outputs and {n_theta} theta outputs from policy_probe.")

    # Infer L_out from a theta sample
    theta_sample = np.asarray(theta_outs[0])
    if theta_sample.ndim != 2:
        raise RuntimeError(f"Unexpected theta output shape: {theta_sample.shape}")
    L_out = theta_sample.shape[1]

    # We want SAME number of steps for all policy components.
    # Use all phi steps, and if there are fewer theta steps, pad the rest with zeros.
    if n_theta < n_phi:
        print(f"⚠️  phi/theta count mismatch (phi={n_phi}, theta={n_theta}); "
              f"padding missing theta steps with zeros to match phi.")
        for _ in range(n_phi - n_theta):
            theta_outs.append(np.zeros((num_paths, L_out), dtype=np.float32))
    elif n_theta > n_phi:
        print(f"⚠️  phi/theta count mismatch (phi={n_phi}, theta={n_theta}); "
              f"truncating extra theta steps to match phi.")
        theta_outs = theta_outs[:n_phi]

    # Now we have exactly the same number of steps for φ and θ
    n_steps = n_phi

    # avg_phi_per_step: (n_steps,)
    avg_phi_per_step = np.empty(n_steps, dtype=np.float64)
    # avg_theta_per_step: (n_steps, L_out)
    avg_theta_per_step = np.empty((n_steps, L_out), dtype=np.float64)

    for t in range(n_steps):
        # phi_outs[t]: (num_paths, 1)
        avg_phi_per_step[t] = phi_outs[t].mean(axis=0)[0]
        # theta_outs[t]: (num_paths, L_out)
        avg_theta_per_step[t, :] = theta_outs[t].mean(axis=0)

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
        f.attrs["n_policy_steps"] = n_steps
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