# util_lob.py

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _cumsum_inv(q, quantity):
    s = 0.0
    for i in range(q.shape[0]):
        s += q[i]
        if s > quantity:
            return i
    return q.shape[0]

@njit(cache=True, fastmath=True)
def _left_pad_trim(q, k):
    n = q.shape[0]
    out = np.zeros(n, dtype=q.dtype)
    end = n - k
    if end > 0:
        for i in range(end):
            out[k + i] = q[i]
    return out

@njit(cache=True, fastmath=True)
def _right_pad_trim(q, k, bar):
    n = q.shape[0]
    out = np.empty(n, dtype=q.dtype)
    if k == 0:
        for i in range(n):
            out[i] = q[i]
        return out
    if k >= n:
        for i in range(n):
            out[i] = bar
        return out
    for i in range(n - k):
        out[i] = q[i + k]
    for i in range(n - k, n):
        out[i] = bar
    return out

@njit(cache=True, fastmath=True)
def _remove_liquidity(q, quantity):
    n = q.shape[0]
    out = np.empty(n, dtype=q.dtype)
    s = 0.0
    for i in range(n):
        s += q[i]
        left = s - quantity
        if left < 0.0:
            left = 0.0
        out[i] = q[i] if q[i] < left else left
    return out

@njit(cache=True, fastmath=True)
def _insert_liquidity(q, quantity, rel_price):
    n = q.shape[0]
    if rel_price < 0 or rel_price >= n or quantity <= 0.0:
        out = np.empty(n, dtype=q.dtype)
        for i in range(n):
            out[i] = q[i]
        return out
    out = np.empty(n, dtype=q.dtype)
    for i in range(n):
        out[i] = q[i]
    out[rel_price] = out[rel_price] + quantity
    return out

@njit(cache=True, fastmath=True)
def _cancel_liquidity(q, quantity, rel_price):
    n = q.shape[0]
    out = np.empty(n, dtype=q.dtype)
    for i in range(n):
        out[i] = q[i]
    if rel_price < 0 or rel_price >= n or quantity <= 0.0:
        return out
    amt = quantity if quantity < out[rel_price] else out[rel_price]
    out[rel_price] = out[rel_price] - amt
    return out

@njit(cache=True, fastmath=True)
def depleted_liquidity(queues, quantity):
    """
    Given per-level quantities 'queues' and target 'quantity',
    return how much executes at each level (prefix-sum clamp).
    """
    n = queues.shape[0]
    out = np.zeros(n, dtype=queues.dtype)
    remaining = quantity
    for i in range(n):
        if remaining <= 0.0:
            break
        take = queues[i] if queues[i] < remaining else remaining
        out[i] = take
        remaining -= take
    return out

@njit(cache=True, fastmath=True)
def bid_ladder(best_ask_price, tick, n_levels):
    """
    Absolute bid execution prices (highest bid first):
      p[i] = best_ask_price - (i+1)*tick, i=0..n_levels-1
    """
    out = np.empty(n_levels, dtype=np.float32)
    p = best_ask_price - tick
    for i in range(n_levels):
        out[i] = p
        p -= tick
    return out

@njit(cache=True, fastmath=True)
def ask_ladder(best_bid_price, tick, n_levels):
    """
    Absolute ask execution prices (lowest ask first):
      p[i] = best_bid_price + (i+1)*tick
    """
    out = np.empty(n_levels, dtype=np.float32)
    p = best_bid_price + tick
    for i in range(n_levels):
        out[i] = p
        p += tick
    return out