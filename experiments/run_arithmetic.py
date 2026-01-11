import numpy as np
from datasets.arithmetic import generate_arithmetic_dataset
from scripts.init_weights import init_trm_weights
from core.decode import decode_answer
from core.reasoning import recursive_reason_loop


def main() -> None:
    # NOTE: Random weights won't "solve" addition; this is a plumbing sanity check.
    D = 32
    K = 19  # vocab = sums 0..18
    n = 6

    weights = init_trm_weights(D, K, seed=0)

    # Vocab is just one-hot for 0..18 in K-dim
    vocab = np.eye(K, dtype=np.float32)

    dataset = generate_arithmetic_dataset(n_samples=10, dim=D, seed=123)

    correct = 0
    for i, (x, y_true, z_init) in enumerate(dataset):
        y0 = np.zeros(K, dtype=np.float32)

        z, y = recursive_reason_loop(x, y0, z_init, n, **weights)
        pred = decode_answer(y, vocab)

        true = int(np.argmax(y_true[:K]))  # y_true is D-dim; true class in first 19 dims by construction
        # if D != K, true label still in y_true indices; we used D=32 so OK.
        # BUT y_true lives in D, and decode vocab is K, so we compute true from y_true[0:K]
        correct += int(pred == true)

        print(f"[{i}] a+b true={true} pred={pred}")

    print(f"\nAccuracy (random weights sanity): {correct}/{len(dataset)}")


if __name__ == "__main__":
    main()
