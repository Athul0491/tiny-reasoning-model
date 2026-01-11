import numpy as np
from typing import List, Tuple
from .model import recursive_latent_update


def recursive_reason_loop(
    x: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    n: int,
    W1: np.ndarray,
    b1: np.ndarray,
    W2_z: np.ndarray,
    b2_z: np.ndarray,
    W2_y: np.ndarray,
    b2_y: np.ndarray,
    tol: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full TRM recursion process:
    - n latent reasoning steps (update z)
    - 1 answer refinement step (update y)
    - optional early stop on y change
    """
    y = y0.astype(np.float32).copy()
    z = z0.astype(np.float32).copy()

    # n latent reasoning steps: update z only (y held constant during these)
    for _ in range(n):
        # We only need z_next here; y returned but we ignore to match paper’s split.
        z, _ = recursive_latent_update(x, y, z, W1, b1, W2_z, b2_z, W2_y, b2_y)

    # 1 answer refinement step: update y with x=0 using the same shared net
    y_prev = y
    x_zero = np.zeros_like(x, dtype=np.float32)
    # Use the same step function by passing x_zero and current z
    _, y_next = recursive_latent_update(x_zero, y, z, W1, b1, W2_z, b2_z, W2_y, b2_y)

    if np.linalg.norm(y_next - y_prev) < tol:
        return z, y_next

    return z, y_next


def supervised_reasoning_loop(
    x: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    n: int,
    T: int,
    W1: np.ndarray,
    b1: np.ndarray,
    W2_z: np.ndarray,
    b2_z: np.ndarray,
    W2_y: np.ndarray,
    b2_y: np.ndarray,
) -> List[np.ndarray]:
    """
    Deep supervision loop for TRM.
    Returns [y0, y1, ..., yT], where each y_t is after one recursion process.
    """
    y = y0.astype(np.float32).copy()
    z = z0.astype(np.float32).copy()

    y_history: List[np.ndarray] = [y.copy()]

    for _ in range(T):
        z, y = recursive_reason_loop(x, y, z, n, W1, b1, W2_z, b2_z, W2_y, b2_y)
        y_history.append(y.copy())

    return y_history


def reasoning_pipeline(
    x: np.ndarray,
    vocab_embeddings: np.ndarray,
    n: int,
    W1: np.ndarray,
    b1: np.ndarray,
    W2_z: np.ndarray,
    b2_z: np.ndarray,
    W2_y: np.ndarray,
    b2_y: np.ndarray,
) -> int:
    """
    init -> latent recursion -> answer refinement -> decode (argmax over vocab dot-products)
    """
    D = int(x.shape[0])
    K = int(vocab_embeddings.shape[1])

    y0 = np.zeros(K, dtype=np.float32)
    z0 = np.zeros(D, dtype=np.float32)

    z, y = recursive_reason_loop(x, y0, z0, n, W1, b1, W2_z, b2_z, W2_y, b2_y)

    scores = vocab_embeddings @ y  # (V,)
    return int(np.argmax(scores))
