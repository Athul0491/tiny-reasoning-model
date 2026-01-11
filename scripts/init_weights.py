import numpy as np
from typing import Dict


def init_trm_weights(D: int, K: int, seed: int = 0, scale: float = 0.02) -> Dict[str, np.ndarray]:
    """
    Deterministic random init for TRM-style 2-layer net.

    Shapes:
      W1: (D, 2D + K)
      b1: (D,)
      W2_z: (D, D), b2_z: (D,)
      W2_y: (K, D), b2_y: (K,)
    """
    rng = np.random.default_rng(seed)

    W1 = (rng.standard_normal((D, 2 * D + K)) * scale).astype(np.float32)
    b1 = (rng.standard_normal((D,)) * scale).astype(np.float32)

    W2_z = (rng.standard_normal((D, D)) * scale).astype(np.float32)
    b2_z = (rng.standard_normal((D,)) * scale).astype(np.float32)

    W2_y = (rng.standard_normal((K, D)) * scale).astype(np.float32)
    b2_y = (rng.standard_normal((K,)) * scale).astype(np.float32)

    return {
        "W1": W1,
        "b1": b1,
        "W2_z": W2_z,
        "b2_z": b2_z,
        "W2_y": W2_y,
        "b2_y": b2_y,
    }
