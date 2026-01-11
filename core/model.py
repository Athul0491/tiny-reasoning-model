import numpy as np
from typing import Tuple


def recursive_latent_update(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2_z: np.ndarray,
    b2_z: np.ndarray,
    W2_y: np.ndarray,
    b2_y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One TRM recursive step:
    1) latent reasoning: z_next = net(x, y, z)
    2) answer refinement: y_next = net(0, y, z_next)
    """

    # ----- Latent reasoning -----
    input_z = np.concatenate([x, y, z], axis=0).astype(np.float32)  # (2D + K,)
    h1_z = np.tanh(W1 @ input_z + b1)  # (D,)
    z_next = (W2_z @ h1_z + b2_z).astype(np.float32)  # (D,)

    # ----- Answer refinement -----
    x_zero = np.zeros_like(x, dtype=np.float32)
    input_y = np.concatenate([x_zero, y, z_next], axis=0).astype(np.float32)
    h1_y = np.tanh(W1 @ input_y + b1)
    y_next = (W2_y @ h1_y + b2_y).astype(np.float32)  # (K,)

    return z_next, y_next
