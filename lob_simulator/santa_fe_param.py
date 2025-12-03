# santa_fe_param.py


# Expected CSV path for a given STOCK symbol:
#   <repo_root>/data/<STOCK>/<STOCK}_santa_fe_param.csv
#
# Example for TSLA:
#   C:\deep_hedging_VWAP\data\TSLA\TSLA_santa_fe_param.csv

import os
import pathlib
import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# 1) Choose stock
# -------------------------------------------------------------------
# You can override this via environment variable LOB_STOCK, e.g.
#   set LOB_STOCK=TSLA
STOCK = os.getenv("LOB_STOCK", "TSLA").upper()


# -------------------------------------------------------------------
# 2) CSV loader + sanity checks (shared by all stocks)
# -------------------------------------------------------------------
def _load_santa_fe_csv(stock: str) -> dict:
    """
    Load Santa Fe parameters for a given stock from its CSV file,
    run sanity checks, and return a dict of parameters.

    All stocks must follow the same CSV schema as TSLA_santa_fe_param.csv.
    """
    stock = stock.upper()

    # Assume structure: <repo_root>/data/<STOCK>/<STOCK>_santa_fe_parame.csv
    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parent  # lob_simulator/.. → repo root
    csv_path = repo_root / "data" / stock / f"{stock}_santa_fe_param.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Santa Fe CSV not found for {stock}: {csv_path}")

    df = pd.read_csv(csv_path)

    # The first column holds parameter names; the 2nd..end are numeric values.
    name_col = df.columns[0]
    value_cols = df.columns[1:]

    def _row_values(name: str) -> np.ndarray:
        """Return numeric values for the row whose first column == name."""
        row = df.loc[df[name_col] == name, value_cols]
        if row.empty:
            raise ValueError(f"Row '{name}' not found in {csv_path}")
        vals = row.to_numpy()[0]
        return vals[~np.isnan(vals)]

    # tick is encoded as the *header* of the first numeric column (e.g. "0.01")
    tick     = float(value_cols[0])

    # Scalars
    K        = int(_row_values("K")[0])
    K_trunc  = int(_row_values("K_trunc")[0])
    L        = int(_row_values("L")[0])
    Q_0      = float(_row_values("Q_0")[0])
    gamma    = float(_row_values("gamma")[0])
    c_infty  = float(_row_values("c^infty")[0])

    mu_M     = float(_row_values("mu^M")[0])
    sigma_M  = float(_row_values("sigma^M")[0])
    mu_L     = float(_row_values("mu^L")[0])
    sigma_L  = float(_row_values("sigma^L")[0])
    mu_C     = float(_row_values("mu^C")[0])
    sigma_C  = float(_row_values("sigma^C")[0])

    S_0         = float(_row_values("S_0")[0])
    varepsilon_0 = float(_row_values("varepsilon_0")[0])

    # Vectors (should all be length K)
    lambda_vec = _row_values("lambda_vec")
    rho_vec    = _row_values("rho_vec")
    a_0_vec    = _row_values("a_0_vec")
    b_0_vec    = _row_values("b_0_vec")

    # ----------------------------------------------------------------
    # Sanity checks (same for all stocks)
    # ----------------------------------------------------------------
    assert K_trunc <= K, f"[{stock}] K_trunc ({K_trunc}) must be ≤ K ({K})"
    assert L <= K_trunc, f"[{stock}] L ({L}) must be ≤ K_trunc ({K_trunc})"

    assert lambda_vec.size == K, f"[{stock}] lambda_vec length {lambda_vec.size} != K ({K})"
    assert rho_vec.size    == K, f"[{stock}] rho_vec length {rho_vec.size} != K ({K})"
    assert a_0_vec.size    == K, f"[{stock}] a_0_vec length {a_0_vec.size} != K ({K})"
    assert b_0_vec.size    == K, f"[{stock}] b_0_vec length {b_0_vec.size} != K ({K})"

    # If we reach here, CSV is in good shape
    return dict(
        STOCK=stock,
        tick=tick,
        K=K,
        K_trunc=K_trunc,
        L=L,
        Q_0=Q_0,
        gamma=gamma,
        c_infty=c_infty,
        mu_M=mu_M,
        sigma_M=sigma_M,
        mu_L=mu_L,
        sigma_L=sigma_L,
        mu_C=mu_C,
        sigma_C=sigma_C,
        S_0=S_0,
        varepsilon_0=varepsilon_0,
        lambda_vec=lambda_vec.astype(float),
        rho_vec=rho_vec.astype(float),
        a_0_vec=a_0_vec.astype(float),
        b_0_vec=b_0_vec.astype(float),
    )


# -------------------------------------------------------------------
# 3) Load the active stock's params and export them as module globals
# -------------------------------------------------------------------
_params = _load_santa_fe_csv(STOCK)

# Inject into module namespace
globals().update(_params)

# Shared simulation / training discretization (can be changed if needed)
time_window = 1.0   # hours