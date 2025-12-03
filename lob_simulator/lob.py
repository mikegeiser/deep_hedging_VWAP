# lob.py

import numpy as np
from numba import njit

# Support both:
#  - import lob_simulator.lob        (package style, used by extract_market_features)
#  - running simulate_events.py flat (from inside lob_simulator)
try:
    # package-relative import
    from .util_lob import (
        _cumsum_inv,
        _left_pad_trim,
        _right_pad_trim,
        _remove_liquidity,
        _insert_liquidity,
        _cancel_liquidity,
    )
except ImportError:
    # fallback for "flat" script usage: python simulate_events.py from lob_simulator
    from util_lob import (
        _cumsum_inv,
        _left_pad_trim,
        _right_pad_trim,
        _remove_liquidity,
        _insert_liquidity,
        _cancel_liquidity,
    )


@njit(cache=True, fastmath=True)
def _execute_mo_sell(S, varepsilon, a_vec, b_vec, qty, tick):
    if qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    iS = _cumsum_inv(b_vec, 0.0)
    iq = _cumsum_inv(b_vec, qty)
    k = iq - iS
    new_S = S - 0.5*k*tick
    new_varepsilon = varepsilon + k*tick
    new_a_vec = _left_pad_trim(a_vec, k)
    new_b_vec = _remove_liquidity(b_vec, qty)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def _execute_mo_buy(S, varepsilon, a_vec, b_vec, qty, tick):
    if qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    iS = _cumsum_inv(a_vec, 0.0)
    iq = _cumsum_inv(a_vec, qty)
    k = iq - iS
    new_S = S + 0.5*k*tick
    new_varepsilon = varepsilon + k*tick
    new_a_vec = _remove_liquidity(a_vec, qty)
    new_b_vec = _left_pad_trim(b_vec, k)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def _submit_lo_sell(S, varepsilon, a_vec, b_vec, qty, rel_k, tick, c_infty):
    n = a_vec.shape[0]
    if rel_k < 0 or rel_k >= n or qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    ask_idx = _cumsum_inv(a_vec, 0.0)
    k = ask_idx - rel_k
    if k < 0:
        k = 0
    new_S = S - 0.5*k*tick
    new_varepsilon = varepsilon - k*tick
    new_a_vec = _insert_liquidity(a_vec, qty, rel_k)
    new_b_vec = _right_pad_trim(b_vec, k, c_infty)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def _submit_lo_buy(S, varepsilon, a_vec, b_vec, qty, rel_k, tick, c_infty):
    n = a_vec.shape[0]
    if rel_k < 0 or rel_k >= n or qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    bid_idx = _cumsum_inv(b_vec, 0.0)
    k = bid_idx - rel_k
    if k < 0:
        k = 0
    new_S = S + 0.5*k*tick
    new_varepsilon = varepsilon - k*tick
    new_a_vec = _right_pad_trim(a_vec, k, c_infty)
    new_b_vec = _insert_liquidity(b_vec, qty, rel_k)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def _cancel_lo_sell(S, varepsilon, a_vec, b_vec, qty, rel_k, tick):
    n = a_vec.shape[0]
    if rel_k < 0 or rel_k >= n or qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    eff = qty if qty < a_vec[rel_k] else a_vec[rel_k]
    ask_idx = _cumsum_inv(a_vec, 0.0)
    if rel_k == ask_idx and qty >= a_vec[rel_k]:
        iS = _cumsum_inv(a_vec, 0.0)
        iq = _cumsum_inv(a_vec, eff)
        k = iq - iS
    else:
        k = 0
    new_S = S + 0.5*k*tick
    new_varepsilon = varepsilon + k*tick
    new_a_vec = _cancel_liquidity(a_vec, eff, rel_k)
    new_b_vec = _left_pad_trim(b_vec, k)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def _cancel_lo_buy(S, varepsilon, a_vec, b_vec, qty, rel_k, tick):
    n = a_vec.shape[0]
    if rel_k < 0 or rel_k >= n or qty <= 0.0:
        return S, varepsilon, a_vec, b_vec

    eff = qty if qty < b_vec[rel_k] else b_vec[rel_k]
    bid_idx = _cumsum_inv(b_vec, 0.0)
    if rel_k == bid_idx and qty >= b_vec[rel_k]:
        iS = _cumsum_inv(b_vec, 0.0)
        iq = _cumsum_inv(b_vec, eff)
        k = iq - iS
    else:
        k = 0
    new_S = S - 0.5*k*tick
    new_varepsilon = varepsilon + k*tick
    new_a_vec = _left_pad_trim(a_vec, k)
    new_b_vec = _cancel_liquidity(b_vec, eff, rel_k)

    return new_S, new_varepsilon, new_a_vec, new_b_vec

@njit(cache=True, fastmath=True)
def apply_event_nb(event, S, varepsilon, a_vec, b_vec, tick, c_infty):
    dt, et, qty, rk = event
    event_type = int(et)
    rel_k = int(rk)

    if event_type == -1:
        return S, varepsilon, a_vec, b_vec
    elif event_type == 0:
        return _execute_mo_sell(S, varepsilon, a_vec, b_vec, qty, tick)
    elif event_type == 1:
        return _execute_mo_buy(S, varepsilon, a_vec, b_vec, qty, tick)
    elif event_type == 2:
        return _submit_lo_sell(S, varepsilon, a_vec, b_vec, qty, rel_k, tick, c_infty)
    elif event_type == 3:
        return _submit_lo_buy(S, varepsilon, a_vec, b_vec, qty, rel_k, tick, c_infty)
    elif event_type == 4:
        return _cancel_lo_sell(S, varepsilon, a_vec, b_vec, qty, rel_k, tick)
    else:  # 5
        return _cancel_lo_buy(S, varepsilon, a_vec, b_vec, qty, rel_k, tick)