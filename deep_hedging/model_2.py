# model_2.py  (restored logic from old model_2.py, adapted to new layout)

import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Concatenate, Layer
from tensorflow.keras.models import Model

from nn_architecture import build_hedging_networks
from deep_hedging_param import N, num_layers, num_neurons, num_outputs
from lob_simulator.santa_fe_param import K_trunc, L, tick, Q_0


# ---------- small utilities ----------

def zeros_like(x, name=None):
    """Shape-safe zeros: avoids Keras Lambda shape inference issues."""
    return Lambda(lambda t: t * 0.0, name=name)(x)


def slice_cols(x, start, width, name=None):
    return Lambda(lambda t: t[:, start:start + width], name=name)(x)


# ---------- Trainable scalar τ (not used, but kept for compatibility) ----------

class TrainableScalar(Layer):
    def __init__(self, init_value=0.0, name="tau", **kwargs):
        super().__init__(name=name, **kwargs)
        self.init_value = float(init_value)

    def build(self, input_shape):
        # scalar; will be broadcast later
        self.scalar = self.add_weight(
            name=f"{self.name}_var",
            shape=(),
            dtype=self.dtype if self.dtype else tf.float32,
            initializer=tf.keras.initializers.Constant(self.init_value),
            trainable=True,
        )

    def call(self, ref_tensor):
        # Broadcast to (batch,1) using the batch dimension of ref_tensor
        batch = tf.shape(ref_tensor)[0]
        return tf.fill([batch, 1], tf.cast(self.scalar, ref_tensor.dtype))


def build_model_hedge_strat(
    K_trunc_in: int,
    L_in: int,
    N_in: int,
    tick_in: float,
    Q0_in: float,
    num_layers_in: int,
):
    """
    Build the hedging model for arbitrary K_trunc and L (vectorized, no hard-coded dims).

    State layout per time step (matching your TSLA_market_features.h5):
        [ mid, spread,
          supply[:K_trunc], demand[:K_trunc],
          betas[:L],
          cur_vol, cur_val ]
    where K_trunc_in = L_in in your current setup.
    """

    # ---- indices in state ----
    STATE_DIM = 2 + K_trunc_in + K_trunc_in + L_in + 2
    IDX_MID = 0
    IDX_SPR = 1
    IDX_SUP = 2
    IDX_DEM = IDX_SUP + K_trunc_in
    IDX_BET = IDX_DEM + K_trunc_in
    IDX_VOL = IDX_BET + L_in
    IDX_VAL = IDX_BET + L_in + 1

    # ---- network layers (per step) ----
    # New nn_architecture API: we must pass N, num_layers, num_neurons, num_outputs
    layers = build_hedging_networks(
        N_in,
        num_layers_in,
        num_neurons,
        num_outputs,
    )  # expected: (num_layers_in+1) * N_in Dense-like layers

    # ---- inputs & step-0 parsing ----
    state = Input(shape=(STATE_DIM,), name="state_0")
    Inputs = [state]

    mid     = slice_cols(state, IDX_MID, 1, "mid_0")
    spread  = slice_cols(state, IDX_SPR, 1, "spread_0")
    supply  = slice_cols(state, IDX_SUP, K_trunc_in, "supply_0")   # (B, K)
    demand  = slice_cols(state, IDX_DEM, K_trunc_in, "demand_0")   # (B, K)
    betas   = slice_cols(state, IDX_BET, L_in, "betas_0")          # (B, L)
    cur_vol = slice_cols(state, IDX_VOL, 1, "cur_vol_0")
    cur_val = slice_cols(state, IDX_VAL, 1, "cur_val_0")

    # ---- carry variables ----
    prev_phi   = zeros_like(mid,   name="prev_phi_init")   # (B,1)
    prev_theta = zeros_like(betas, name="prev_theta_init") # (B,L)
    delta      = zeros_like(mid,   name="delta_init")      # (B,1)
    stoch_int  = zeros_like(mid,   name="stoch_int_init")  # (B,1)
    cost       = zeros_like(mid,   name="cost_init")       # (B,1)
    cum_volume = cur_vol
    cum_value  = cur_val

    # ---- price ladders (constant steps) ----
    ask_steps = tf.reshape(tf.range(1, K_trunc_in + 1, dtype=tf.float32),
                           (1, K_trunc_in))
    bid_steps = tf.reshape(tf.range(1, L_in + 1, dtype=tf.float32),
                           (1, L_in))

    # ---- unrolled time steps ----
    for n in range(N_in):
        # policy input
        FNN_in = Concatenate()(
            [mid, spread, supply, demand, betas,
             cum_volume, cum_value, prev_phi, prev_theta]
        )

        FNN = FNN_in
        for i in range(num_layers_in + 1):
            FNN = layers[i + n * (num_layers_in + 1)](FNN)

        # outputs: phi (B,1) + thetas (B,L)
        phi = Lambda(lambda x: x[:, :1], name=f"phi_{n}")(FNN)
        if n < N_in - 1:
            thetas = Lambda(lambda x: x[:, 1:1 + L_in], name=f"thetas_{n}")(FNN)
        else:
            # last step: no limit orders, consistent with your old setup
            thetas = Lambda(lambda x: x[:, 1:1 + L_in] * 0.0,
                            name=f"thetas_{n}")(FNN)

        # -------- executed qty from phi vs supplies --------
        csum      = Lambda(lambda s: tf.cumsum(s, axis=1),
                           name=f"csum_{n}")(supply)                             # (B,K)
        pre_csum  = Lambda(
            lambda c: tf.concat([tf.zeros_like(c[:, :1]), c[:, :-1]], axis=1),
            name=f"pre_csum_{n}"
        )(csum)                                                                 # (B,K)
        remaining = Lambda(
            lambda xs: tf.nn.relu(xs[0] - xs[1]),
            name=f"remaining_{n}"
        )([phi, pre_csum])                                                      # (B,K)
        fill_phi  = Lambda(
            lambda xs: tf.minimum(xs[0], xs[1]),
            name=f"fill_phi_{n}"
        )([supply, remaining])                                                  # (B,K)
        phi_exec  = Lambda(
            lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
            name=f"phi_exec_{n}"
        )(fill_phi)                                                             # (B,1)

        # -------- executed qty from previous LOs --------
        if betas.shape[-1] is not None:
            assert betas.shape[-1] == L_in, \
                f"betas last dim {betas.shape[-1]} != L {L_in}"
        thetas_exec = Lambda(
            lambda xs: tf.reduce_sum(xs[0] * xs[1], axis=1, keepdims=True),
            name=f"thetas_exec_{n}"
        )([betas, prev_theta])                                                  # (B,1)

        # inventory delta accumulation
        delta = Lambda(
            lambda xs: xs[0] + xs[1] + xs[2],
            name=f"delta_acc_{n}"
        )([delta, phi_exec, thetas_exec])

        # -------- price ladders per step (reconstruct best quotes from mid & spread) --------
        curr_bid = Lambda(
            lambda xs: xs[0] - 0.5 * xs[1],
            name=f"bid_p_{n}"
        )([mid, spread])                                                        # (B,1)
        curr_ask = Lambda(
            lambda xs: xs[0] + 0.5 * xs[1],
            name=f"ask_p_{n}"
        )([mid, spread])                                                        # (B,1)
        ask_prices_vec = Lambda(
            lambda xs: xs[0] + tick_in * xs[1],
            name=f"ask_prices_vec_{n}"
        )([curr_bid, ask_steps])                                                # (B,K)
        bid_prices_vec = Lambda(
            lambda xs: xs[0] - tick_in * xs[1],
            name=f"bid_prices_vec_{n}"
        )([curr_ask, bid_steps])                                                # (B,L)

        # -------- trading cost --------
        cost_phi = Lambda(
            lambda xs: tf.reduce_sum(xs[0] * (xs[1] - xs[2]),
                                     axis=1, keepdims=True),
            name=f"cost_phi_{n}"
        )([fill_phi, ask_prices_vec, mid])

        cost_theta = Lambda(
            lambda xs: tf.reduce_sum(xs[0] * xs[1] * (xs[2] - xs[3]),
                                     axis=1, keepdims=True),
            name=f"cost_theta_{n}"
        )([betas, prev_theta, bid_prices_vec, mid])

        cost = Lambda(
            lambda xs: xs[0] + xs[1] + xs[2],
            name=f"cost_acc_{n}"
        )([cost, cost_phi, cost_theta])

        # -------- next state --------
        state = Input(shape=(STATE_DIM,), name=f"state_{n+1}")
        Inputs += [state]

        next_mid = slice_cols(state, IDX_MID, 1, f"mid_{n+1}")
        spread   = slice_cols(state, IDX_SPR, 1, f"spread_{n+1}")
        supply   = slice_cols(state, IDX_SUP, K_trunc_in, f"supply_{n+1}")
        demand   = slice_cols(state, IDX_DEM, K_trunc_in, f"demand_{n+1}")
        betas    = slice_cols(state, IDX_BET, L_in, f"betas_{n+1}")
        cur_vol  = slice_cols(state, IDX_VOL, 1, f"cur_vol_{n+1}")
        cur_val  = slice_cols(state, IDX_VAL, 1, f"cur_val_{n+1}")

        # -------- trading profit --------
        price_diff = Lambda(
            lambda xs: xs[0] - xs[1],
            name=f"price_diff_{n}"
        )([next_mid, mid])
        stoch_int  = Lambda(
            lambda xs: xs[0] + xs[1] * xs[2],
            name=f"stoch_int_acc_{n}"
        )([stoch_int, delta, price_diff])

        # -------- values for VWAP --------
        cum_volume = Lambda(
            lambda xs: xs[0] + xs[1],
            name=f"cum_volume_acc_{n}"
        )([cum_volume, cur_vol])
        cum_value  = Lambda(
            lambda xs: xs[0] + xs[1],
            name=f"cum_value_acc_{n}"
        )([cum_value,  cur_val])

        # -------- update for next step --------
        mid = next_mid
        prev_phi   = phi
        prev_theta = thetas

    # ---- final metrics ----
    VWAP      = Lambda(
        lambda xs: xs[0] / (xs[1] + 1e-6),
        name="VWAP"
    )([cum_value, cum_volume])
    liability = Lambda(
        lambda xs: Q0_in * (xs[0] - xs[1]),
        name="liability"
    )([mid, VWAP])

    over        = Lambda(lambda d: tf.abs(d - Q0_in), name="dev_over")(delta)
    half_spread = Lambda(lambda s: 0.5 * s, name="half_spread_final")(spread)
    dev_pen     = Lambda(
        lambda xs: xs[0] * xs[1],
        name="deviation_penalty"
    )([over, half_spread])

    wealth = Lambda(
        lambda xs: xs[0] - (xs[1] + xs[2] + xs[3]),
        name="wealth"
    )([stoch_int, cost, liability, dev_pen])  # (B,1)

    # If you ever want RU with trainable τ inside the model:
    # tau_broadcast = TrainableScalar(init_value=5000.0, name="tau")(wealth)
    # ru_out = Concatenate(name="ru_out")([wealth, tau_broadcast])

    model = Model(inputs=Inputs, outputs=wealth, name="hedge_model")
    return model


# ------- build default model with your global params -------

model_hedge_strat = build_model_hedge_strat(
    K_trunc_in=K_trunc,
    L_in=L,
    N_in=N,
    tick_in=tick,
    Q0_in=Q_0,
    num_layers_in=num_layers,
)

# Probes still work exactly as before
phi_outs = [model_hedge_strat.get_layer(f"phi_{n}").output for n in range(N)]

theta_outs = []
for n in range(N):
    try:
        theta_outs.append(model_hedge_strat.get_layer(f"thetas_{n}").output)
    except ValueError:
        break

policy_probe = Model(
    inputs=model_hedge_strat.inputs,
    outputs=phi_outs + theta_outs,
    name="policy_probe"
)

try:
    final_delta_tensor = model_hedge_strat.get_layer(f"delta_acc_{N-1}").output
except ValueError:
    final_delta_tensor = None
    for n in range(N - 1, -1, -1):
        try:
            final_delta_tensor = model_hedge_strat.get_layer(f"delta_acc_{n}").output
            break
        except ValueError:
            continue
    if final_delta_tensor is None:
        raise RuntimeError("Could not locate any 'delta_acc_*' layer to build delta_probe.")

delta_probe = Model(
    inputs=model_hedge_strat.inputs,
    outputs=final_delta_tensor,
    name="delta_probe"
)

mid_outs = []
for n in range(N + 1):
    try:
        mid_outs.append(model_hedge_strat.get_layer(f"mid_{n}").output)
    except ValueError:
        break
mid_probe = Model(
    inputs=model_hedge_strat.inputs,
    outputs=mid_outs,
    name="mid_probe"
)
