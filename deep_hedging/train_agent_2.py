# train_agent_2.py — mini batch, RU (global tau) + (optional) gradient accumulation)

import os
import sys
import json
import h5py
import argparse
import warnings
import subprocess
import numpy as np
from pathlib import Path

# --- Quiet TensorFlow/TFP logs (set BEFORE importing tensorflow) ---
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
warnings.filterwarnings("ignore", category=UserWarning, module="keras")
warnings.filterwarnings("ignore", category=UserWarning, module="tf_keras")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow_probability")

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# --- project params (new module) ---
from deep_hedging_param import (
    N,
    epochs,
    batch_size,
    learning_rate,
    early_stopping_patience,
    lr_schedule_factor,
    lr_schedule_patience,
    min_lr,
)

# --- loss (new API) ---
from loss_function import RU_objective

# --- model: now using model_2 instead of model ---
from model_2 import model_hedge_strat


# ------------- newline-safe print (avoids clashing with Keras progbar) -------------
def _p(msg: str):
    print(f"\n{msg}", flush=True)


# ----------------------------- Sound when training finishes -----------------------------
def _ding(wav=None):
    try:
        if sys.platform.startswith("win"):
            import winsound
            if wav and os.path.exists(wav):
                winsound.PlaySound(wav, winsound.SND_FILENAME)
            else:
                try:
                    winsound.PlaySound("SystemNotification", winsound.SND_ALIAS)
                except Exception:
                    winsound.Beep(880, 300)
                    winsound.Beep(660, 300)
                    winsound.Beep(440, 500)
        elif sys.platform == "darwin":  # macOS
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:  # Linux
            for cmd in (
                ["paplay", "/usr/share/sounds/freedesktop/stereo/complete.oga"],
                ["aplay", "/usr/share/sounds/alsa/Front_Center.wav"],
            ):
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    break
                except Exception:
                    pass
            else:
                print("\a", end="", flush=True)  # terminal bell as last resort
    except Exception:
        print("\a", end="", flush=True)


class SoundOnFinish(tf.keras.callbacks.Callback):
    def __init__(self, wav=None):
        super().__init__()
        self.wav = wav

    def on_train_end(self, logs=None):
        _ding(self.wav)


# ----------------------------- Epoch loss reporter (improvement + clean printing) -----------------------------
class EpochLossReporter(tf.keras.callbacks.Callback):
    """
    Prints, on its own line:
      - If improved: "Improved loss: prev → current (Δ=..., Δ%=...) — saving model weights"
      - Else:        "No improvement: current (best stays ...)"
    Mirrors save_best_only(min) on 'loss'.
    """

    def __init__(self, monitor="loss", mode="min", min_delta=0.0):
        super().__init__()
        self.monitor = monitor
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode
        self.min_delta = float(min_delta)
        self.best = None  # start unknown

    def _is_improvement(self, current, best):
        if best is None:
            return True
        if self.mode == "min":
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def on_train_begin(self, logs=None):
        self.best = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
        try:
            current = float(current)
        except Exception:
            return

        if self.best is None:
            self.best = current
            _p(f"Initial {self.monitor}: {current:,.4f} — saving model weights")
            return

        if self._is_improvement(current, self.best):
            prev = self.best
            delta = current - prev
            pct = (delta / abs(prev)) * 100.0 if prev != 0 else float("nan")
            self.best = current
            _p(
                f"Improved {self.monitor}: {prev:,.4f} → {current:,.4f} "
                f"(Δ={delta:,.4f}, {pct:+.4f}%) — saving model weights"
            )
        else:
            _p(
                f"No improvement: {self.monitor}={current:,.4f} "
                f"(best stays {self.best:,.4f})"
            )


# ----------------------------- Diagnostics: count optimizer updates -----------------------------
class CountOptimizerUpdates(tf.keras.callbacks.Callback):
    """Print how many optimizer.apply_gradients() happen per epoch (useful with accumulation)."""

    def on_epoch_begin(self, epoch, logs=None):
        self._start_iter = int(self.model.optimizer.iterations.numpy())

    def on_epoch_end(self, epoch, logs=None):
        end_iter = int(self.model.optimizer.iterations.numpy())
        num_updates = end_iter - self._start_iter
        _p(f"[ACCUM] epoch {epoch+1}: optimizer.apply_gradients calls = {num_updates}")


# ----------------------------- JSON logger (writes every epoch) -----------------------------
class JSONLogger(tf.keras.callbacks.Callback):
    """
    Append/merge epoch logs into a JSON file at each epoch end.
    Uses atomic replace to avoid partial writes on crash.
    """

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.store = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.store = json.load(f)
            except Exception:
                try:
                    os.replace(path, path + ".bak")
                except Exception:
                    pass
                self.store = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        clean = {}
        for k, v in logs.items():
            try:
                clean[k] = float(v)
            except Exception:
                clean[k] = v if isinstance(v, (int, float)) else str(v)
        for k, v in clean.items():
            self.store.setdefault(k, []).append(v)

        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.store, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)


# ----------------------------- LR logger (ensures LR is in logs) -----------------------------
class LRLogger(tf.keras.callbacks.Callback):
    """Inject current learning rate into logs at epoch end."""

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        logs["learning_rate"] = lr


# ----------------------------- RU: global τ updater -----------------------------
class UpdateGlobalTau(tf.keras.callbacks.Callback):
    def __init__(self, x_full, y_full, tau_var, alpha=0.99, pred_bs=8192):
        super().__init__()
        self.x_full = x_full
        self.y_full = y_full
        self.tau_var = tau_var
        self.alpha = float(alpha)
        self.pred_bs = int(pred_bs)

    def on_train_begin(self, logs=None):
        self._refresh_tau()

    def on_epoch_begin(self, epoch, logs=None):
        self._refresh_tau()

    def _refresh_tau(self):
        y_pred = self.model.predict(self.x_full, batch_size=self.pred_bs, verbose=0)
        z = -(y_pred - self.y_full).reshape(-1)
        try:
            q = float(np.quantile(z, self.alpha, method="linear"))
        except TypeError:
            q = float(np.percentile(z, self.alpha * 100.0, interpolation="linear"))
        self.tau_var.assign(q)


class AssertTauConstant(tf.keras.callbacks.Callback):
    def __init__(self, tau_var, atol=0.0, rtol=0.0, verbose=True):
        super().__init__()
        self.tau_var = tau_var
        self.atol = float(atol)
        self.rtol = float(rtol)
        self.verbose = verbose
        self.epoch_tau = None

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_tau = float(self.tau_var.numpy())
        if self.verbose:
            print(f"[τ] epoch {epoch+1} start: tau = {self.epoch_tau:.6f}")

    def on_train_batch_begin(self, batch, logs=None):
        t = float(self.tau_var.numpy())
        import numpy as np

        if not np.isclose(t, self.epoch_tau, rtol=self.rtol, atol=self.atol):
            raise RuntimeError(
                f"τ changed within epoch: was {self.epoch_tau}, now {t} at batch {batch}"
            )


# ----------------------------- Stateful EarlyStopping -----------------------------
class StatefulEarlyStopping(EarlyStopping):
    """
    EarlyStopping that reconstructs best + wait from previous history and,
    on train end, restores weights from an external best checkpoint file.
    """

    def __init__(self, hist_store=None, initial_epoch=0, best_weights_path=None, **kwargs):
        self.hist_store = hist_store or {}
        self.initial_epoch = int(initial_epoch or 0)
        self.best_weights_path = best_weights_path
        super().__init__(**kwargs)

    def _is_improvement(self, current, best):
        if best is None:
            return True

        mode = self.mode
        if mode == "auto":
            if ("acc" in self.monitor) or ("auc" in self.monitor) or self.monitor.startswith(
                "fmeasure"
            ):
                mode = "max"
            else:
                mode = "min"

        if mode == "min":
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        if self.initial_epoch <= 0:
            return
        if self.monitor not in self.hist_store:
            return

        values = self.hist_store[self.monitor][: self.initial_epoch]
        if not values:
            return

        best = None
        wait = 0

        for v in values:
            try:
                current = float(v)
            except Exception:
                continue
            if np.isnan(current):
                continue

            if best is None:
                best = current
                wait = 0
                continue

            if self._is_improvement(current, best):
                best = current
                wait = 0
            else:
                wait += 1

        if best is None:
            return

        self.best = best
        self.wait = wait
        self.stopped_epoch = 0

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Epoch {self.stopped_epoch + 1}: early stopping")

        if (
            self.restore_best_weights
            and self.best_weights_path
            and os.path.exists(self.best_weights_path)
        ):
            try:
                self.model.load_weights(self.best_weights_path)
                if self.verbose > 0:
                    print("Restored model weights from best checkpoint file.")
            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not restore best weights from file: {e}")


# ----------------------------- Stateful ReduceLROnPlateau -----------------------------
class StatefulReduceLROnPlateau(ReduceLROnPlateau):
    """
    ReduceLROnPlateau that reconstructs best, wait, cooldown_counter from history
    so patience/cooldown behavior continues seamlessly after resume.
    """

    def __init__(self, hist_store=None, initial_epoch=0, **kwargs):
        self.hist_store = hist_store or {}
        self.initial_epoch = int(initial_epoch or 0)
        super().__init__(**kwargs)

    def _is_improvement(self, current, best):
        if best is None:
            return True

        mode = self.mode
        if mode == "auto":
            if ("acc" in self.monitor) or ("auc" in self.monitor) or self.monitor.startswith(
                "fmeasure"
            ):
                mode = "max"
            else:
                mode = "min"

        if mode == "min":
            return (best - current) > self.min_delta
        else:
            return (current - best) > self.min_delta

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        if self.initial_epoch <= 0:
            return
        if self.monitor not in self.hist_store:
            return

        values = self.hist_store[self.monitor][: self.initial_epoch]
        if not values:
            return

        best = None
        wait = 0
        cooldown_counter = 0

        for v in values:
            try:
                current = float(v)
            except Exception:
                continue
            if np.isnan(current):
                continue

            if cooldown_counter > 0:
                cooldown_counter -= 1
                wait = 0

            if best is None:
                best = current
                wait = 0
                continue

            if self._is_improvement(current, best):
                best = current
                wait = 0
            else:
                if cooldown_counter == 0:
                    wait += 1
                    if wait >= self.patience:
                        cooldown_counter = self.cooldown
                        wait = 0

        if best is None:
            return

        self.best = best
        self.wait = wait
        self.cooldown_counter = cooldown_counter


# ----------------------------- Main script -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Train deep hedging model on <symbol>_market_features.h5 features."
    )
    ap.add_argument(
        "-s", "--symbol", "--stock",
        dest="symbol",
        required=True,
        help="Stock / ticker symbol, e.g. TSLA, AAPL, CSCO.",
    )
    ap.add_argument(
        "--base-dir",
        default="",
        help="Project base directory. If empty, defaults to repo root.",
    )
    ap.add_argument(
        "--ready-h5",
        default="",
        help=(
            "Path to <symbol>_market_features.h5 "
            "(defaults to <base-dir>/data/<symbol>/<symbol>_market_features.h5)."
        ),
    )
    ap.add_argument(
        "--stem",
        default="",
        help="Label for outputs (defaults to '<symbol>_market_features').",
    )
    ap.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="Micro-batches to accumulate before one optimizer step.",
    )
    args = ap.parse_args()

    symbol = args.symbol.strip().upper()

    # ---- repo root (one level above this file's folder)
    this_file = Path(__file__).resolve()
    repo_root = this_file.parents[1]  # e.g. C:\deep_hedging_VWAP

    # ---- Base dir: either CLI or repo root
    base_dir = args.base_dir.strip() or str(repo_root)

    # ---- Derive defaults
    stem = args.stem.strip() or f"{symbol}_market_features"
    ready_h5 = args.ready_h5.strip() or os.path.join(
        base_dir, "data", symbol, f"{symbol}_market_features.h5"
    )

    # Save dir under repo_root/data/<symbol>
    save_dir = os.path.join(str(repo_root), "data", symbol)
    os.makedirs(save_dir, exist_ok=True)

    best_weights_path = os.path.join(save_dir, f"{stem}_best.weights.h5")
    last_weights_path = os.path.join(save_dir, f"{stem}_last.weights.h5")
    history_path = os.path.join(save_dir, f"{stem}_history.json")

    # ======= Load Training Data =======
    if not os.path.exists(ready_h5):
        raise FileNotFoundError(f"Ready H5 file not found: {ready_h5}")

    with h5py.File(ready_h5, "r") as f:
        if "snapshots" not in f:
            raise KeyError(f"Dataset 'snapshots' not found in {ready_h5}.")
        feats = f["snapshots"][:]  # (num_paths, num_snaps, F)
        num_paths, num_snaps, F = feats.shape

        f_symbol = f.attrs.get("symbol", "").strip().upper()
        if f_symbol and f_symbol != symbol:
            print(
                f"⚠️  Symbol in file attrs = '{f_symbol}' differs from CLI symbol '{symbol}'. Proceeding anyway."
            )

        # Try to read K_trunc/L from attrs; if missing, infer from F
        K_trunc_attr = int(f.attrs.get("K_trunc", 0))
        L_attr = int(f.attrs.get("L", 0))

        if K_trunc_attr == 0 or L_attr == 0:
            if (F - 4) % 3 != 0:
                raise ValueError(
                    f"Cannot infer L from F={F}; expected (F-4) divisible by 3."
                )
            L = (F - 4) // 3
            K_trunc = L
            print(
                f"ℹ️  K_trunc / L attrs missing; inferred L={L}, K_trunc={K_trunc} from F={F}."
            )
        else:
            K_trunc = K_trunc_attr
            L = L_attr
            expected_F = 2 * K_trunc + L + 4
            if expected_F != F:
                raise ValueError(
                    f"Feature width mismatch: F={F}, expected {expected_F} (=2*K_trunc+L+4)."
                )

        if num_snaps != (N + 1):
            raise ValueError(
                f"Snapshot count mismatch: file has {num_snaps}, but N+1 = {N+1}."
            )

    print(
        f"✅ Loaded features: paths={num_paths}, snaps={num_snaps}, F={F} from {ready_h5}"
    )

    # ======= DEBUG: inspect one path of raw features before building x_train =======
    def _fmt(x, width=10, prec=4):
        try:
            return f"{float(x):{width}.{prec}f}"
        except Exception:
            return f"{x!r:>{width}}"

    inspect_path_id = 2  # change this to inspect another path index

    if 0 <= inspect_path_id < num_paths:
        print("\n" + "=" * 80)
        print(f"🔍 DEBUG: Inspecting raw features for path_id = {inspect_path_id}")
        print(f"  shape feats = (paths={num_paths}, snaps={num_snaps}, F={F})")
        print(f"  L = {L}, K_trunc = {K_trunc}")

        # Guess layout based on F, L, K_trunc
        if F == (3 * L + 4):
            layout = "market_features"   # [S, eps, a[:L], b[:L], e[:L], vol, val]
        elif F == (2 * K_trunc + L + 4):
            layout = "training_features" # [mid, spread, a[:K], b[:K], betas[:L], vol, val]
        else:
            layout = "unknown_raw"

        print(f"  Guessed layout: {layout}")
        print("=" * 80)

        path_data = feats[inspect_path_id]  # shape (num_snaps, F)

        if layout == "market_features":
            idx_S = 0
            idx_eps = 1
            idx_a_start = 2
            idx_a_end = idx_a_start + L
            idx_b_start = idx_a_end
            idx_b_end = idx_b_start + L
            idx_e_start = idx_b_end
            idx_e_end = idx_e_start + L
            idx_vol = F - 2
            idx_val = F - 1

            for t in range(num_snaps):
                row = path_data[t]
                S = float(row[idx_S])
                eps = float(row[idx_eps])
                ask0 = S + 0.5 * eps
                bid0 = S - 0.5 * eps
                a_vec = row[idx_a_start:idx_a_end]
                b_vec = row[idx_b_start:idx_b_end]
                e_vec = row[idx_e_start:idx_e_end]
                vol = float(row[idx_vol])
                val = float(row[idx_val])

                print(
                    f"\n[t={t}]  S={_fmt(S)}, eps={_fmt(eps)}, "
                    f"best_bid={_fmt(bid0)}, best_ask={_fmt(ask0)}, "
                    f"vol={_fmt(vol)}, val={_fmt(val)}"
                )
                print(f"{'lvl':>3} | {'a_vec':>12} | {'b_vec':>12} | {'e_full':>7}")
                print("-" * 46)
                for k in range(L):
                    av = a_vec[k]
                    bv = b_vec[k]
                    ev = e_vec[k]
                    print(
                        f"{k:>3} | {_fmt(av, width=12, prec=4)} | "
                        f"{_fmt(bv, width=12, prec=4)} | {_fmt(ev, width=7, prec=2)}"
                    )
            print("=" * 80)

        elif layout == "training_features":
            # [mid, spread, a[:K_trunc], b[:K_trunc], betas[:L], vol, val]
            idx_mid = 0
            idx_spread = 1
            idx_a_start = 2
            idx_a_end = idx_a_start + K_trunc
            idx_b_start = idx_a_end
            idx_b_end = idx_b_start + K_trunc
            idx_beta_start = idx_b_end
            idx_beta_end = idx_beta_start + L
            idx_vol = F - 2
            idx_val = F - 1

            for t in range(num_snaps):
                row = path_data[t]
                mid = float(row[idx_mid])
                spread = float(row[idx_spread])
                a_vec = row[idx_a_start:idx_a_end]
                b_vec = row[idx_b_start:idx_b_end]
                betas = row[idx_beta_start:idx_beta_end]
                vol = float(row[idx_vol])
                val = float(row[idx_val])

                print(
                    f"\n[t={t}]  mid={_fmt(mid)}, spread={_fmt(spread)}, "
                    f"vol={_fmt(vol)}, val={_fmt(val)}"
                )
                print(f"{'lvl':>3} | {'a_vec':>12} | {'b_vec':>12} | {'beta':>12}")
                print("-" * 50)
                for k in range(K_trunc):
                    av = a_vec[k]
                    bv = b_vec[k]
                    be = betas[k] if k < L else 0.0
                    print(
                        f"{k:>3} | {_fmt(av, width=12, prec=4)} | "
                        f"{_fmt(bv, width=12, prec=4)} | {_fmt(be, width=12, prec=4)}"
                    )
            print("=" * 80)

        else:
            # unknown layout, dump raw vectors
            for t in range(num_snaps):
                print(f"\n[t={t}] raw feature vector:")
                print(path_data[t])
            print("=" * 80)

    # ======= Build x_train / y_train =======
    # x_train: list of (num_paths, F) for t=0..N
    x_train = [feats[:, t, :].astype(np.float32) for t in range(num_snaps)]
    y_train = np.zeros((num_paths, 1), dtype=np.float32)

    # ======= Effective micro-batch size =======
    accum_steps = int(max(1, args.accum_steps))
    micro_batch_size = max(1, batch_size // accum_steps)
    total_steps = int(np.ceil(len(y_train) / micro_batch_size))
    expected_updates = (
        int(np.ceil(total_steps / max(2, accum_steps)))
        if accum_steps >= 2
        else total_steps
    )

    if accum_steps >= 2:
        print(
            f"[ACCUM] accum_steps={accum_steps}, micro_batch_size={micro_batch_size} "
            f"(effective ~ {micro_batch_size * accum_steps}); "
            f"steps/epoch≈{total_steps}, expected optimizer updates/epoch≈{expected_updates}"
        )
    else:
        print(
            f"[ACCUM] disabled (accum_steps=1); batch_size={micro_batch_size}; steps/epoch≈{total_steps}"
        )

    # ======= Compile =======
    tau_var = tf.Variable(0.0, dtype=tf.float32, trainable=False, name="global_tau")
    loss_fn = RU_objective(tau_var)

    if accum_steps >= 2:
        try:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=learning_rate, gradient_accumulation_steps=accum_steps
            )
        except TypeError:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            if hasattr(optimizer, "gradient_accumulation_steps"):
                optimizer.gradient_accumulation_steps = accum_steps
            else:
                print(
                    "[ACCUM] Built-in gradient accumulation not supported in this tf.keras version.\n"
                    "        You will get one optimizer update per micro-batch.\n"
                    "        (Upgrade Keras or switch to a custom accumulation wrapper.)"
                )
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    tf.get_logger().setLevel("ERROR")
    model_hedge_strat.compile(optimizer=optimizer, loss=loss_fn)

    # ======= Resume from checkpoint if history/weights exist =======
    initial_epoch = 0
    hist_store = {}
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                hist_store = json.load(f)

            if "loss" in hist_store and len(hist_store["loss"]) > 0:
                initial_epoch = len(hist_store["loss"])
                print(
                    f"🔁 Resuming training from epoch {initial_epoch} (history has {initial_epoch} epochs)."
                )

                if os.path.exists(last_weights_path):
                    model_hedge_strat.load_weights(last_weights_path)
                    print(f"🔁 Loaded last (current) weights from: {last_weights_path}")
                elif os.path.exists(best_weights_path):
                    model_hedge_strat.load_weights(best_weights_path)
                    print(
                        f"🔁 Last weights not found; loaded best weights from: {best_weights_path}"
                    )
                else:
                    print(
                        "⚠️ History found but no weights file found; starting from scratch weights."
                    )

                last_lr = None
                if "learning_rate" in hist_store and len(hist_store["learning_rate"]) >= initial_epoch:
                    last_lr = float(hist_store["learning_rate"][initial_epoch - 1])
                elif "lr" in hist_store and len(hist_store["lr"]) >= initial_epoch:
                    last_lr = float(hist_store["lr"][initial_epoch - 1])

                if last_lr is not None:
                    try:
                        lr_obj = model_hedge_strat.optimizer.learning_rate
                        if isinstance(lr_obj, tf.Variable) or hasattr(lr_obj, "assign"):
                            lr_obj.assign(last_lr)
                        else:
                            model_hedge_strat.optimizer.learning_rate = last_lr
                        print(f"🔁 Restored optimizer learning rate to {last_lr}")
                    except Exception as e:
                        print(f"⚠️ Could not restore learning rate ({e})")
            else:
                print(
                    "ℹ️ History file exists but 'loss' is empty or missing; starting from scratch."
                )
        except Exception as e:
            print(f"⚠️ Could not read/parse history; starting from scratch. Error: {e}")
            hist_store = {}
            initial_epoch = 0
    else:
        print("ℹ️ No history file found; starting training from scratch.")

    # ======= Callbacks =======
    tau_cb = UpdateGlobalTau(
        x_full=x_train,
        y_full=y_train,
        tau_var=tau_var,
        alpha=0.99,
        pred_bs=max(1024, micro_batch_size),
    )
    assert_tau = AssertTauConstant(tau_var, atol=0.0, rtol=0.0, verbose=True)
    early_stopping = StatefulEarlyStopping(
        hist_store=hist_store,
        initial_epoch=initial_epoch,
        best_weights_path=best_weights_path,
        monitor="loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint_best = ModelCheckpoint(
        filepath=best_weights_path,
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=0,
    )
    checkpoint_last = ModelCheckpoint(
        filepath=last_weights_path,
        monitor="loss",
        save_best_only=False,
        save_weights_only=True,
        verbose=0,
    )
    reporter = EpochLossReporter(monitor="loss", mode="min", min_delta=0.0)
    lr_schedule = StatefulReduceLROnPlateau(
        hist_store=hist_store,
        initial_epoch=initial_epoch,
        monitor="loss",
        factor=lr_schedule_factor,
        patience=lr_schedule_patience,
        min_lr=min_lr,
        verbose=1,
    )
    updates_counter = CountOptimizerUpdates()
    lr_logger = LRLogger()
    json_cb = JSONLogger(history_path)
    sound_cb = SoundOnFinish(wav=r"C:\Windows\Media\tada.wav")

    history = model_hedge_strat.fit(
        x=x_train,
        y=y_train,
        batch_size=micro_batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        shuffle=True,
        verbose=1,
        callbacks=[
            tau_cb,
            assert_tau,
            early_stopping,
            reporter,
            checkpoint_best,
            checkpoint_last,
            lr_schedule,
            updates_counter,
            lr_logger,
            json_cb,
            sound_cb,
        ],
    )

    try:
        with open(history_path, "w") as f:
            json.dump(json_cb.store, f)
        print("Final history saved")
    except Exception:
        pass

    print("Saved best and last model weights")


if __name__ == "__main__":
    main()
