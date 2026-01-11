import numpy as np
from scripts.init_weights import init_trm_weights
from core.reasoning import reasoning_pipeline


def test_reasoning_pipeline_runs():
    D, K = 32, 16
    w = init_trm_weights(D, K, seed=0)

    x = np.random.randn(D).astype(np.float32)
    vocab = np.random.randn(10, K).astype(np.float32)

    pred = reasoning_pipeline(x, vocab, n=6, **w)
    assert isinstance(pred, int)
    assert 0 <= pred < vocab.shape[0]
