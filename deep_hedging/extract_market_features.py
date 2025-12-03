# extract_market_features.py

import argparse
import gc
from time import time
from pathlib import Path
import sys

import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm

# Make sure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deep_hedging_param import N
from lob_simulator.santa_fe_param import _load_santa_fe_csv, time_window, L
from lob_simulator.lob import apply_event_nb
from lob_simulator.util_lob import depleted_liquidity, bid_ladder, ask_ladder


def reconstruct_one_path(events, params):
    """
    Reconstruct LOB state from a sequence of events and extract features
    on a regular time grid.

    Parameters
    ----------
    events : np.ndarray, shape (N_events,), structured dtype with fields
        'dt'   : float16
        'size' : float16
        'type' : int8
        'rel'  : int16 (or int8 for older files)
    params : dict
        Output of _load_santa_fe_csv(stock), containing:
        K, S_0, varepsilon_0, a_0_vec, b_0_vec, c_infty, tick, time_window, ...

    Returns
    -------
    out : np.ndarray, shape (N+1, D), dtype float32
        Per-time-step market features:
          0      : mid price S_t
          1      : spread ε_t
          2..    : a_0..a_{L-1}
          ...    : b_0..b_{L-1}
          ...    : e_0..e_{L-1} (fill ratios)
          last-1 : vol (traded volume in (t_n, t_{n+1}])
          last   : val (trade value in (t_n, t_{n+1}])
    """
    K = int(params["K"])
    S = float(params["S_0"])
    varepsilon = float(params["varepsilon_0"])
    a_vec = params["a_0_vec"].astype(np.float64).copy()
    b_vec = params["b_0_vec"].astype(np.float64).copy()
    c_infty = params["c_infty"]
    tick = params["tick"]

    # Global time horizon
    T = 60 * 60 * time_window

    # Feature dimension: first L queues for ask/bid + L e's + 2 (S, ε) + 2 (vol, val)
    D = 3 * L + 4

    # Output array (float32)
    out = np.empty((N + 1, D), dtype=np.float32)

    def best_prices(S_val, varepsilon_val):
        ask_price = S_val + 0.5 * varepsilon_val
        bid_price = S_val - 0.5 * varepsilon_val
        return ask_price, bid_price

    # Initial snapshot at t = 0
    ask_price, bid_price = best_prices(S, varepsilon)
    out[0, 0] = S
    out[0, 1] = varepsilon
    out[0, 2: 2 + L] = a_vec[:L]
    out[0, 2 + L: 2 + 2 * L] = b_vec[:L]
    out[0, 2 + 2 * L: 2 + 3 * L] = 0.0  # e's
    out[0, 2 + 3 * L] = 0.0             # vol
    out[0, 2 + 3 * L + 1] = 0.0         # val

    # Time grid
    t_grid = np.linspace(0.0, T, N + 1)

    # Absolute event times
    if events.shape[0] > 0:
        dt = events["dt"].astype(np.float32)
        t_events = np.cumsum(dt)
    else:
        dt = np.zeros(0, dtype=np.float32)
        t_events = np.zeros(0, dtype=np.float32)
    n_events = events.shape[0]

    idx = 0  # pointer into events

    for n in range(N):
        t_start = t_grid[n]
        t_end = t_grid[n + 1]

        vol = 0.0
        val = 0.0
        min_bid_hit = np.inf
        min_ask_seen = np.inf

        # Precompute initial absolute bid prices & ticks for this step
        ask_price_step_start, _ = best_prices(S, varepsilon)
        initial_abs_bid_prices = bid_ladder(ask_price_step_start, tick, K)  # shape (K,)
        init_ticks = np.rint(initial_abs_bid_prices / tick).astype(np.int64)

        # Process events in (t_start, t_end]
        while idx < n_events and t_events[idx] <= t_end:
            et = int(events[idx]["type"])
            qty = float(events[idx]["size"])
            rel = int(events[idx]["rel"])

            # Best prices before event
            ask_price_curr, bid_price_curr = best_prices(S, varepsilon)

            if et == 0:  # Sell MO hits the bid
                vol += qty
                exctn_prcs = bid_ladder(ask_price_curr, tick, b_vec.shape[0])
                exctn_qntts = depleted_liquidity(b_vec, qty)
                val += float(np.dot(exctn_prcs, exctn_qntts))

                mask = exctn_qntts > 0
                if np.any(mask):
                    hit_px = exctn_prcs[mask][-1]
                    if hit_px < min_bid_hit:
                        min_bid_hit = hit_px

            elif et == 1:  # Buy MO lifts the ask
                vol += qty
                exctn_prcs = ask_ladder(bid_price_curr, tick, a_vec.shape[0])
                exctn_qntts = depleted_liquidity(a_vec, qty)
                val += float(np.dot(exctn_prcs, exctn_qntts))

            # Update LOB state
            S, varepsilon, a_vec, b_vec = apply_event_nb(
                (float(dt[idx]), et, qty, rel),
                S, varepsilon, a_vec, b_vec,
                tick, c_infty
            )

            # Track min best ask seen during the step
            ask_price_after, _ = best_prices(S, varepsilon)
            if ask_price_after < min_ask_seen:
                min_ask_seen = ask_price_after

            idx += 1

        # Compute fill ratios (for K levels, then slice to L)
        reference_price = min(min_bid_hit, min_ask_seen)
        if np.isfinite(reference_price):
            ref_ticks = int(np.rint(reference_price / tick))
            e_full = (
                (init_ticks > ref_ticks).astype(np.float32) * 1.0 +
                (init_ticks == ref_ticks).astype(np.float32) * 0.5
            ).astype(np.float32)
        else:
            e_full = np.zeros(K, dtype=np.float32)

        # Snapshot at end of step n -> row n+1
        ask_price, bid_price = best_prices(S, varepsilon)
        row = n + 1
        out[row, 0] = S
        out[row, 1] = varepsilon
        out[row, 2: 2 + L] = a_vec[:L]
        out[row, 2 + L: 2 + 2 * L] = b_vec[:L]
        out[row, 2 + 2 * L: 2 + 3 * L] = e_full[:L]
        out[row, 2 + 3 * L] = vol
        out[row, 2 + 3 * L + 1] = val

        # If no more events, fill remaining rows with constant state and zero flows
        if idx >= n_events:
            for m in range(n + 1, N):
                row2 = m + 1
                out[row2, 0] = S
                out[row2, 1] = varepsilon
                out[row2, 2: 2 + L] = a_vec[:L]
                out[row2, 2 + L: 2 + 2 * L] = b_vec[:L]
                out[row2, 2 + 2 * L: 2 + 3 * L] = 0.0
                out[row2, 2 + 3 * L] = 0.0
                out[row2, 2 + 3 * L + 1] = 0.0
            break

    return out


def reconstruct_one_path_from_file(path_index, events_path_str, params):
    """
    Worker helper: open the HDF5 events file, load one path by index,
    and run reconstruct_one_path(events, params).
    """
    with h5py.File(events_path_str, "r") as fev:
        events_ds = fev["events"]
        events = events_ds[path_index]  # 1D array (N_events,) with structured dtype
    return reconstruct_one_path(events, params)


def save_snapshots_h5(file_path, path_data_list, N_val, D_val):
    """
    Append a batch of snapshot paths (each shape: (N+1, D)) to an H5 file.
    Stored as float32 to avoid overflow in trade value.
    """
    path_data_array = np.stack(path_data_list, axis=0).astype(np.float32)  # (batch_size, N+1, D)

    with h5py.File(file_path, "a") as f:
        if "snapshots" not in f:
            ds = f.create_dataset(
                "snapshots",
                data=path_data_array,
                maxshape=(None, N_val + 1, D_val),
                dtype="float32",
                compression="lzf",
            )
            ds.attrs["N"] = N_val
            ds.attrs["D"] = D_val
            ds.attrs["L"] = L
        else:
            ds = f["snapshots"]
            current_paths = ds.shape[0]
            new_paths = current_paths + path_data_array.shape[0]
            ds.resize((new_paths, N_val + 1, D_val))
            ds[current_paths:] = path_data_array


def main():
    parser = argparse.ArgumentParser(
        description="Extract regular-grid market features from simulated events."
    )
    parser.add_argument("--stock", type=str, required=True, help="Stock symbol, e.g. TSLA, CSCO, INTC, PCLN.")
    parser.add_argument(
        "--events-path",
        type=str,
        default=None,
        help="HDF5 file with event sequences (output of simulate_events.py). "
             "If not provided, defaults to ../data/<STOCK>/<STOCK>_events.h5",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="HDF5 file to write market features to. "
             "If not provided, defaults to ../data/<STOCK>/<STOCK>_market_features.h5",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for each chunk.")
    parser.add_argument("--chunk-size", type=int, default=100, help="How many paths per chunk to process & save.")
    args = parser.parse_args()

    stock = args.stock.upper()

    # Load params for this stock (for K, initial state, etc.)
    params = _load_santa_fe_csv(stock)
    K = int(params["K"])
    D = 3 * L + 4

    # Determine events path
    if args.events_path is None:
        here = Path(__file__).resolve().parent  # deep_hedging/
        repo_root = here.parent
        events_path = repo_root / "data" / stock / f"{stock}_events.h5"
    else:
        events_path = Path(args.events_path)

    # Determine save path
    if args.save_path is None:
        here = Path(__file__).resolve().parent
        repo_root = here.parent
        save_path = repo_root / "data" / stock / f"{stock}_market_features.h5"
    else:
        save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    # How many paths in events file?
    with h5py.File(events_path, "r") as fev:
        if "events" not in fev:
            raise KeyError(f"No 'events' dataset found in {events_path}")
        n_paths_total = fev["events"].shape[0]

    print(
        f"[~] Stock: {stock} | Source: {events_path} | Target market features: {save_path}\n"
        f"    N_paths = {n_paths_total}, N = {N}, K = {K}, L = {L}, D = {D}"
    )

    t0 = time()
    remaining = n_paths_total
    processed = 0
    chunk_idx = 0

    while remaining > 0:
        chunk_idx += 1
        batch = min(args.chunk_size, remaining)
        start_idx = processed
        end_idx = processed + batch

        desc = f"Chunk {chunk_idx} (paths {start_idx}..{end_idx-1})"

        # Reconstruct in parallel by index; each worker opens the HDF5 file
        results = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            prefer="processes",
            pre_dispatch="n_jobs",
            batch_size=1,
        )(
            delayed(reconstruct_one_path_from_file)(i, str(events_path), params)
            for i in tqdm(range(start_idx, end_idx), desc=desc)
        )

        save_snapshots_h5(str(save_path), results, N, D)
        del results
        gc.collect()

        processed += batch
        remaining -= batch

        with h5py.File(save_path, "r") as fsnap:
            total_saved = fsnap["snapshots"].shape[0]
        print(f"✓ Saved chunk {chunk_idx}: {batch} paths (total in file: {total_saved})")

    print(f"✅ Done. Wrote market features for {processed} paths to {save_path}")
    print(f"⏱️ Elapsed: {round(time() - t0, 2)} seconds")


if __name__ == "__main__":
    main()
