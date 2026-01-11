import numpy as np
from typing import Any, Dict, List
from .reasoning import recursive_reason_loop


def run_ablation_study(
    configs: List[Dict[str, Any]],
    x: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    vocab: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2_z: np.ndarray,
    b2_z: np.ndarray,
    W2_y: np.ndarray,
    b2_y: np.ndarray,
) -> Dict[int, Any]:
    """
    Runs ablation experiments across different recursion configs.
    Config keys:
      - n: latent reasoning steps per recursion process (default 6)
      - T: number of recursion processes (deep supervision steps) (default 1)
      - tol: optional convergence tolerance
    """
    results: Dict[int, Any] = {}

    for idx, cfg in enumerate(configs):
        n = int(cfg.get("n", 6))
        T = int(cfg.get("T", 1))
        tol = float(cfg.get("tol", 1e-5))

        y = y0.astype(np.float32).copy()
        z = z0.astype(np.float32).copy()

        # Run T recursion processes
        for _ in range(T):
            z, y = recursive_reason_loop(x, y, z, n, W1, b1, W2_z, b2_z, W2_y, b2_y, tol=tol)

        scores = vocab @ y
        pred_idx = int(np.argmax(scores))

        results[idx] = {
            "config": cfg,
            "prediction": pred_idx,
            "final_y_norm": float(np.linalg.norm(y)),
            "final_z_norm": float(np.linalg.norm(z)),
        }

    return results
