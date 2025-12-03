# simulate_events.py

import os
import argparse
import gc
from time import time
from pathlib import Path

import numpy as np
import h5py
from joblib import Parallel, delayed
from tqdm import tqdm

from santa_fe_param import _load_santa_fe_csv, time_window
from santa_fe_model import sample_event_nb
from lob import apply_event_nb


# Structured dtype for a single event:
EVENT_DTYPE = np.dtype([
    ("dt",   "<f2"),  # float16 for time increments
    ("size", "<f4"),  # float32 for size to avoid overflow to inf
    ("type", "i1"),   # int8
    ("rel",  "<i2"),  # int16
])


def simulate_one_path_events(params):
    """
    Simulate a single LOB path, returning ONLY the event sequence.

    Parameters
    ----------
    params : dict
        Output of _load_santa_fe_csv(stock), containing:
        K, gamma, lambda_vec, rho_vec, mu_M, sigma_M, mu_L, sigma_L, mu_C,
        sigma_C, S_0, varepsilon_0, a_0_vec, b_0_vec, c_infty, tick.

    Returns
    -------
    events : np.ndarray, shape (N_events,), dtype=EVENT_DTYPE
        Each entry n is:
            (dt_n, size_n, type_n, rel_price_n)
        where dt_n is the time increment from event n-1 to event n.
        - dt   : float16
        - size : float32
        - type : int8
        - rel  : int8
    """
    K = int(params["K"])
    gamma = params["gamma"]
    lambda_vec = params["lambda_vec"]
    rho_vec = params["rho_vec"]
    mu_M = params["mu_M"]
    sigma_M = params["sigma_M"]
    mu_L = params["mu_L"]
    sigma_L = params["sigma_L"]
    mu_C = params["mu_C"]
    sigma_C = params["sigma_C"]
    S_0 = params["S_0"]
    varepsilon_0 = params["varepsilon_0"]
    a_0_vec = params["a_0_vec"]
    b_0_vec = params["b_0_vec"]
    c_infty = params["c_infty"]
    tick = params["tick"]

    # Total simulation time horizon (seconds)
    T = 60 * 60 * time_window

    # Initial state
    S_t = S_0
    varepsilon_t = varepsilon_0
    a_t_vec = a_0_vec.astype(np.float64).copy()
    b_t_vec = b_0_vec.astype(np.float64).copy()
    time_n = 0.0  # time_0 = 0

    events_py = []

    while True:
        # Sample next event based on current state
        dt, event_type, quantity, rel_price = sample_event_nb(
            K,
            gamma, lambda_vec, rho_vec,
            mu_M, sigma_M, mu_L, sigma_L, mu_C, sigma_C,
            S_t, varepsilon_t, a_t_vec, b_t_vec
        )

        time_next = time_n + dt

        # If the next event would exceed T, stop without recording it
        if time_next > T:
            break

        # Store this event as Python scalars
        events_py.append((dt, quantity, int(event_type), int(rel_price)))

        # Apply the event to update the state
        S_t, varepsilon_t, a_t_vec, b_t_vec = apply_event_nb(
            (dt, event_type, quantity, rel_price),
            S_t, varepsilon_t, a_t_vec, b_t_vec,
            tick, c_infty
        )
        time_n = time_next

    if len(events_py) == 0:
        return np.empty((0,), dtype=EVENT_DTYPE)

    out = np.empty(len(events_py), dtype=EVENT_DTYPE)
    for i, (dt, size, et, rp) in enumerate(events_py):
        out[i]["dt"]   = np.float16(dt) # dt stays float16 to save space; typical dt values are small, no overflow risk
        out[i]["size"] = np.float32(size) # size is stored as float32 to avoid float16 overflow to inf
        out[i]["type"] = np.int8(et)
        out[i]["rel"]  = np.int16(rp)

    return out


def save_to_h5_events(file_path, path_data_list):
    """
    Append a batch of event-based paths (ragged over time) to an H5 file.

    Each path in path_data_list is a 1D array of dtype=EVENT_DTYPE, shape (N_events,).

    We store:
      - 'events': variable-length 1D arrays of EVENT_DTYPE (each element is one full path)
      - attribute 'event_dtype' on 'events' (string repr of dtype)
    """
    vlen_events = h5py.vlen_dtype(EVENT_DTYPE)

    with h5py.File(file_path, "a") as f:
        # Create or open 'events'
        if "events" not in f:
            ev_ds = f.create_dataset(
                "events",
                shape=(0,),
                maxshape=(None,),
                dtype=vlen_events,
                compression="lzf",
            )
            ev_ds.attrs["event_dtype"] = str(EVENT_DTYPE)
        else:
            ev_ds = f["events"]
            if "event_dtype" in ev_ds.attrs:
                if ev_ds.attrs["event_dtype"] != str(EVENT_DTYPE):
                    raise ValueError(
                        f"Inconsistent event_dtype: existing {ev_ds.attrs['event_dtype']} vs new {EVENT_DTYPE}"
                    )

        current_paths = ev_ds.shape[0]
        batch_size = len(path_data_list)
        new_paths = current_paths + batch_size

        ev_ds.resize((new_paths,))

        for i, arr in enumerate(path_data_list):
            ev_ds[current_paths + i] = arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=str, required=True, help="Stock symbol, e.g. TSLA, CSCO, INTC, PCLN.")
    parser.add_argument("--num-paths", type=int, required=True, help="Total number of paths to simulate.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for each chunk.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="How many paths per chunk to simulate & save.")
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="HDF5 file to append to. If not provided, defaults to ../data/<STOCK>/<STOCK>_events.h5",
    )
    args = parser.parse_args()

    stock = args.stock.upper()

    # Load Santa Fe params for this stock directly from CSV
    params = _load_santa_fe_csv(stock)
    K = int(params["K"])  # not used here, but keeps logic explicit

    # Determine output path
    if args.save_path is None:
        here = Path(__file__).resolve().parent  # lob_simulator/
        repo_root = here.parent                # project root
        out_dir = repo_root / "data" / stock
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / f"{stock}_events.h5"
    else:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # How many paths already saved?
    if save_path.exists():
        with h5py.File(save_path, "r") as f:
            start_id = f["events"].shape[0] if "events" in f else 0
    else:
        start_id = 0

    total_to_sim = args.num_paths
    print(
        f"[~] Stock: {stock} | Target: {total_to_sim} paths; "
        f"starting at path_id = {start_id}; using {args.n_jobs} workers; chunk={args.chunk_size}"
    )

    t0 = time()
    remaining = total_to_sim
    produced = 0
    chunk_idx = 0

    while remaining > 0:
        chunk_idx += 1
        batch = min(args.chunk_size, remaining)

        desc = f"Chunk {chunk_idx} ({produced + 1}..{produced + batch})"

        # Simulate one chunk in parallel with a nice tqdm progress bar
        results = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            prefer="processes",
            pre_dispatch="n_jobs",
            batch_size=1,
        )(
            delayed(simulate_one_path_events)(params)
            for _ in tqdm(range(batch), desc=desc)
        )

        # Save and free memory
        save_to_h5_events(str(save_path), results)
        del results
        gc.collect()

        produced += batch
        remaining -= batch
        with h5py.File(save_path, "r") as f:
            total_saved = f["events"].shape[0]
        print(f"✓ Saved chunk {chunk_idx}: {batch} paths (total in file: {total_saved})")

    print(f"✅ Done. Saved {produced} new paths to {save_path}")
    print(f"⏱️ Elapsed: {round(time() - t0, 2)} seconds")


if __name__ == "__main__":
    main()
