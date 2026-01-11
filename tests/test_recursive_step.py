import numpy as np
from scripts.init_weights import init_trm_weights
from core.model import recursive_latent_update


def test_recursive_latent_update_shapes():
    D, K = 32, 16
    w = init_trm_weights(D, K, seed=0)

    x = np.random.randn(D).astype(np.float32)
    y = np.random.randn(K).astype(np.float32)
    z = np.random.randn(D).astype(np.float32)

    z_next, y_next = recursive_latent_update(x, y, z, **w)

    assert z_next.shape == (D,)
    assert y_next.shape == (K,)
    assert z_next.dtype == np.float32
    assert y_next.dtype == np.float32
