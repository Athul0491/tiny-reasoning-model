import numpy as np
from scripts.init_weights import init_trm_weights
from core.ablation import run_ablation_study


def main() -> None:
    D = 32
    K = 16
    x = np.random.randn(D).astype(np.float32)
    y0 = np.zeros(K, dtype=np.float32)
    z0 = np.zeros(D, dtype=np.float32)

    # Example vocab (V=64 candidates)
    V = 64
    vocab = np.random.randn(V, K).astype(np.float32)

    weights = init_trm_weights(D, K, seed=7)

    configs = [
        {"n": 2, "T": 1},
        {"n": 4, "T": 1},
        {"n": 6, "T": 1},
        {"n": 6, "T": 3},
        {"n": 8, "T": 3, "tol": 1e-6},
    ]

    results = run_ablation_study(configs, x, y0, z0, vocab, **weights)

    print("Ablation results:")
    for idx, out in results.items():
        cfg = out["config"]
        print(
            f"  [{idx}] cfg={cfg} pred={out['prediction']} "
            f"y_norm={out['final_y_norm']:.4f} z_norm={out['final_z_norm']:.4f}"
        )


if __name__ == "__main__":
    main()
