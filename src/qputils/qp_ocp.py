from .qp import QP
import numpy as np

class QPOCP:
    r"""
    Class for OCP QP problems
    
    \[
    \min \sum ...
    \]

    \[
    x_{n+1} = A_{n}x_{n} + B_{n}u_{n} + b_{n}
    \]

    Parameters:
        R: N x nU x nU
        Q: N x nX x nX
        S: N x nU x nX
        r: N x nU
        q: N x nX
        A: (N-1) x nX x nX
        B: (N-1) x nX x nU
        b: (N-1) x nX
        lbu: N x nU
        ubu: N x nU
        lbx: N x nX
        ubx: N x nX
        C: N x nC x nU
        D: N x nC x nX
        lg: N x nC
        ug: N x nC
    """
    def __init__(self, R: np.ndarray, Q: np.ndarray, S: np.ndarray, r: np.ndarray, q: np.ndarray, A: np.ndarray, B: np.ndarray, b: np.ndarray, lbu: np.ndarray, ubu: np.ndarray, lbx: np.ndarray, ubx: np.ndarray, C: np.ndarray, D: np.ndarray, lg: np.ndarray, ug: np.ndarray):
        N = R.shape[0]
        nU = R.shape[1]
        nX = Q.shape[1]
        nC = C.shape[1]
        assert R.shape == (N, nU, nU)
        assert Q.shape == (N, nX, nX)
        assert S.shape == (N, nU, nX)
        assert r.shape == (N, nU)
        assert q.shape == (N, nX)
        self.R = R
        self.Q = Q
        self.S = S
        self.r = r
        self.q = q
        self.A = A
        self.B = B
        self.b = b
        self.lbu = lbu
        self.ubu = ubu
        self.lbx = lbx
        self.ubx = ubx
        self.C = C
        self.D = D
        self.lg = lg
        self.ug = ug

    def get_horizon(self):
        return self.R.shape[0]
    
    def get_controls_count(self):
        return self.R.shape[1]

    def get_states_count(self):
        return self.Q.shape[1]


    def to_sparse(self) -> QP:
        N = self.get_horizon()
        nX = self.get_states_count()
        nU = self.get_controls_count()
        H_blocks = []
        for Rn, Qn, Sn in zip(self.R, self.Q, self.S):
            H_blocks.append(np.block([
                [Rn, Sn],
                [Sn, Qn],
            ]))
        H = np.block(
            [[np.zeros((i*(nU+nX),nU+nX)), H_block, np.zeros(())] for i, H_block in enumerate(H_blocks)]
        )
        for An, Bn in self.A, self.B:
            np.stack([An, Bn, np.ones((nX,))])
        q = np.array(zip(self.r, self.q)).flatten()
        # TODO
        return QP(H, q)

    def to_dense(self) -> QP:
        return self.condense(1).to_sparse()

    def condense(self, N: int) -> "QPOCP":
        raise NotImplementedError("TODO")