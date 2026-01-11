import numpy as np
from scripts.init_weights import init_trm_weights
from core.reasoning import recursive_reason_loop


def main() -> None:
    D = 32
    K = 16
    n = 6

    w = init_trm_weights(D, K, seed=42)

    x = np.random.randn(D).astype(np.float32)
    y0 = np.zeros(K, dtype=np.float32)
    z0 = np.zeros(D, dtype=np.float32)

    z, y = recursive_reason_loop(x, y0, z0, n, **w)

    print("✅ smoke_test ok")
    print("z shape:", z.shape, "y shape:", y.shape)
    print("z norm:", float(np.linalg.norm(z)), "y norm:", float(np.linalg.norm(y)))


if __name__ == "__main__":
    main()
