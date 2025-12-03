# santa_fe_model.py

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _draw_index_from_probs(probs):
    total = 0.0
    for i in range(probs.shape[0]):
        total += probs[i]
    if total <= 1e-12:
        return 0
    r = np.random.random()
    acc = 0.0
    for i in range(probs.shape[0]):
        acc += probs[i]
        if r <= acc:
            return i
    return probs.shape[0] - 1

@njit(cache=True, fastmath=True)
def sample_event_nb(
    K,
    gamma, lambda_vec, rho_vec,
    mu_M, sigma_M, mu_L, sigma_L, mu_C, sigma_C,
    S_t, varepsilon_t, a_t_vec, b_t_vec
):
    # Aggregate intensities
    lam   = 0.0
    rho_a = 0.0
    rho_b = 0.0
    for i in range(K):
        lam   += lambda_vec[i]
        rho_a += rho_vec[i] * a_t_vec[i]
        rho_b += rho_vec[i] * b_t_vec[i]

    # Total intensity: 2(gamma + lambda) + rho_a + rho_b
    lam_tot = 2.0 * (gamma + lam) + rho_a + rho_b

    # dt ~ Exp(lam_tot)
    u = np.random.random()
    if u <= 1e-16:
        u = 1e-16
    dt = -np.log(u) / lam_tot

    # choose event type
    w0 = gamma
    w1 = w0 + gamma
    w2 = w1 + lam
    w3 = w2 + lam
    w4 = w3 + rho_a
    # total is w4 + rho_b == lam_tot
    r = np.random.random() * lam_tot
    if r < w0:
        et = 0  # sell MO
    elif r < w1:
        et = 1  # buy MO
    elif r < w2:
        et = 2  # sell LO
    elif r < w3:
        et = 3  # buy LO
    elif r < w4:
        et = 4  # cancel sell LO
    else:
        et = 5  # cancel buy LO

    # size ~ lognormal per type
    if et == 0 or et == 1:      # MO
        qty = np.exp(mu_M + sigma_M * np.random.standard_normal())
    elif et == 2 or et == 3:    # LO
        qty = np.exp(mu_L + sigma_L * np.random.standard_normal())
    else:                       # CXL
        qty = np.exp(mu_C + sigma_C * np.random.standard_normal())

    # relative price k for L/C events
    rk = -1

    if et == 2 or et == 3:
        # LOs: normalize lambda_vec
        pL = np.empty(K, dtype=np.float64)
        if lam > 1e-12:
            for i in range(K):
                pL[i] = lambda_vec[i] / lam
        else:
            for i in range(K):
                pL[i] = 0.0
        rk = _draw_index_from_probs(pL)

    elif et == 4:
        # Cancel sell LO: weights ∝ rho_vec * a_t_vec
        pCa = np.empty(K, dtype=np.float64)
        if rho_a > 1e-12:
            for i in range(K):
                pCa[i] = rho_vec[i] * a_t_vec[i] / rho_a
        else:
            for i in range(K):
                pCa[i] = 0.0
        rk = _draw_index_from_probs(pCa)

    elif et == 5:
        # Cancel buy LO: weights ∝ rho_vec * b_t_vec
        pCb = np.empty(K, dtype=np.float64)
        if rho_b > 1e-12:
            for i in range(K):
                pCb[i] = rho_vec[i] * b_t_vec[i] / rho_b
        else:
            for i in range(K):
                pCb[i] = 0.0
        rk = _draw_index_from_probs(pCb)

    return dt, et, qty, rk