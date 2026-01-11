import numpy as np


def decode_answer(y: np.ndarray, vocab_embeddings: np.ndarray) -> int:
    """
    Select best answer index via dot-product similarity.
    """
    scores = vocab_embeddings @ y
    return int(np.argmax(scores))
