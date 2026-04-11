import pytest
import numpy as np
import qputils

def test_zero_eval():
    H = np.zeros((2, 2))
    q = np.zeros((2,))
    qp = qputils.QP(H, q)
    assert qp.evaluate_primal(np.array([1, 2])) == 0

def test_lp_eval():
    H = np.zeros((2, 2))
    q = np.array([1, 2])
    qp = qputils.QP(H, q)
    # 1*3 + 2*4 == 11
    assert qp.evaluate_primal(np.array([3, 4])) == 11