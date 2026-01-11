import numpy as np
from typing import List, Tuple


def generate_arithmetic_dataset(
    n_samples: int,
    dim: int,
    seed: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates n_samples of (x, y_true, z_init).
    x = one_hot(a) + one_hot(b)
    y_true = one_hot(a + b)
    z_init = zeros(dim)

    Notes:
      - sums go from 0..18, so dim must be >= 19
      - seed makes this deterministic
    """
    if dim < 19:
        raise ValueError("dim must be >= 19 to represent sums up to 18")

    rng = np.random.default_rng(seed)
    dataset: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    for _ in range(n_samples):
        a = int(rng.integers(0, 10))
        b = int(rng.integers(0, 10))
        target = a + b

        one_hot_a = np.zeros(dim, dtype=np.float32)
        one_hot_b = np.zeros(dim, dtype=np.float32)
        one_hot_t = np.zeros(dim, dtype=np.float32)

        one_hot_a[a] = 1.0
        one_hot_b[b] = 1.0
        one_hot_t[target] = 1.0

        x = one_hot_a + one_hot_b
        y_true = one_hot_t
        z_init = np.zeros(dim, dtype=np.float32)

        dataset.append((x, y_true, z_init))

    return dataset
