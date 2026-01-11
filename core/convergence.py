import numpy as np


def check_convergence(
    z_prev: np.ndarray,
    z_curr: np.ndarray,
    y_prev: np.ndarray,
    y_curr: np.ndarray,
    tol: float,
) -> bool:
    """
    Converged if both latent and answer change are below tol (L2).
    """
    z_dist = np.linalg.norm(z_curr - z_prev)
    y_dist = np.linalg.norm(y_curr - y_prev)
    return bool((z_dist < tol) and (y_dist < tol))
