import numpy as np
from core.convergence import check_convergence


def test_check_convergence_true():
    a = np.zeros(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    assert check_convergence(a, b, a, b, tol=1e-6) is True


def test_check_convergence_false():
    a = np.zeros(8, dtype=np.float32)
    b = np.ones(8, dtype=np.float32)
    assert check_convergence(a, b, a, b, tol=1e-6) is False
