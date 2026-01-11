import numpy as np
from scripts.init_weights import init_trm_weights
from core.model import recursive_latent_update


def main() -> None:
    D = 32
    K = 16
    weights = init_trm_weights(D, K, seed=123)

    x = np.random.randn(D).astype(np.float32)
    y = np.zeros(K, dtype=np.float32)
    z = np.zeros(D, dtype=np.float32)

    print("Debugging a few TRM steps:")
    for t in range(5):
        z_next, y_next = recursive_latent_update(x, y, z, **weights)
        print(
            f"step={t} | "
            f"||z||={float(np.linalg.norm(z)):.4f} -> {float(np.linalg.norm(z_next)):.4f} | "
            f"||y||={float(np.linalg.norm(y)):.4f} -> {float(np.linalg.norm(y_next)):.4f} | "
            f"dy={float(np.linalg.norm(y_next - y)):.6f}"
        )
        z, y = z_next, y_next


if __name__ == "__main__":
    main()
