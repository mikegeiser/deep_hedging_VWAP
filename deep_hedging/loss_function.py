# loss_function.py

import tensorflow as tf
import tensorflow_probability as tfp


def ES(y_true, y_pred, alpha=0.99):
    """
    Expected Shortfall (ES) loss at level alpha.
    """
    z = -(y_pred - y_true)  # Negative P&L
    z = tf.reshape(z, [-1])  # Flatten
    VaR = tfp.stats.percentile(z, q=alpha * 100.0, interpolation='linear')
    tail_z = tf.boolean_mask(z, z >= VaR)
    return tf.reduce_mean(tail_z)

def RU_objective(tau_var, alpha=0.99):
    """
    Factory for Rockafellar–Uryasev ES loss with a *fixed* global threshold tau.

    Args:
        tau_var: tf.Variable scalar (non-trainable). Must be updated externally
                 (e.g., via a callback) to the *global* VaR for the current epoch.
        alpha:   ES level in (0,1).

    Returns:
        A Keras-compatible loss function that computes:
            tau + (1/(1-alpha)) * mean( relu( z - tau ) ),
        where z = -(y_pred - y_true).
    """
    alpha = float(alpha)
    inv_one_minus_alpha = 1.0 / (1.0 - alpha)

    def loss(y_true, y_pred):
        # losses (e.g., negative P&L); flatten to 1-D
        z = -(y_pred - y_true)
        z = tf.reshape(z, [-1])

        # hold tau fixed (no grad through tau); match dtype
        tau = tf.stop_gradient(tf.cast(tau_var, z.dtype))

        # RU hinge objective
        return tau + inv_one_minus_alpha * tf.reduce_mean(tf.nn.relu(z - tau))

    return loss