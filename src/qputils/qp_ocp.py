from .qp import QP
import numpy as np

class QPOCP:
    r"""
    Class for OCP QP problems
    
    \[
    \min \sum_{t}^{N} \frac{1}{2}x^T_tQ_tx_t + \frac{1}{2}u^T_tR_tu_t + x^T_tS_tu_t + q^Tx_t + r^Tu_t
    \]

    \[
    x_{t+1} = A_{t}x_{t} + B_{t}u_{t}
    \]

    \[
    \begin{equation}
    \begin{split}
    lbx_{t} & \leq & x_t & \leq & ubx_{t} \\
    lbu_{t} & \leq & u_t & \leq & ubu_{t} \\
    lg_{t} & \leq & Cx + Du & \leq & ug_{t}
    \end{split}
    \end{equation}
    \]

    Parameters:
        Q: N x nX x nX
        R: N x nU x nU
        S: N x nU x nX
        q: N x nX
        r: N x nU
        A: (N-1) x nX x nX
        B: (N-1) x nX x nU
        b: (N-1) x nX
        lbu: N x nU
        ubu: N x nU
        lbx: N x nX
        ubx: N x nX
        C: N x nC x nX
        D: N x nC x nU
        lg: N x nC
        ug: N x nC
    """
    def __init__(self, Q: list[np.ndarray], R: list[np.ndarray], S: list[np.ndarray], q: list[np.ndarray], r: list[np.ndarray], A: list[np.ndarray], B: list[np.ndarray], b: list[np.ndarray], lbu: list[np.ndarray], ubu: list[np.ndarray], lbx: list[np.ndarray], ubx: list[np.ndarray], C: list[np.ndarray], D: list[np.ndarray], lg: list[np.ndarray], ug: list[np.ndarray]):
        N = len(Q)
        assert len(Q) == N
        assert len(R) == N
        assert len(S) == N
        assert len(q) == N
        assert len(r) == N
        assert len(A) == N-1
        assert len(B) == N-1
        assert len(b) == N-1
        assert len(lbu) == N
        assert len(ubu) == N
        assert len(lbx) == N
        assert len(ubx) == N
        assert len(C) == N
        assert len(D) == N
        assert len(lg) == N
        assert len(ug) == N
        for Qt, Rt, St, rt, qt, lbut, ubut, lbxt, ubxt, Ct, Dt, lgt, ugt in zip(Q, R, S, r, q, lbu, ubu, lbx, ubx, C, D, lg, ug):
            nU = Rt.shape[0]
            nX = Qt.shape[0]
            nC = Ct.shape[0]
            assert Qt.shape == (nX, nX)
            assert Rt.shape == (nU, nU)
            assert St.shape == (nU, nX)
            assert qt.shape == (nX,)
            assert rt.shape == (nU,)
            assert lbut.shape == (nU,)
            assert ubut.shape == (nU,)
            assert lbxt.shape == (nX,)
            assert ubxt.shape == (nX,)
            assert Ct.shape == (nC, nX)
            assert Dt.shape == (nC, nU)
            assert lgt.shape == (nC,)
            assert ugt.shape == (nC,)
        for t, (At, Bt, bt) in enumerate(zip(self.A, self.B, self.b)):
            nXt0 = Q[t].shape[0]
            nXt1 = Q[t+1].shape[0]
            nUt0 = R[t].shape[0]
            assert At.shape == (nXt1, nXt0)
            assert Bt.shape == (nXt1, nUt0)
            assert bt.shape == (nXt1,)
        self.Q = Q
        self.R = R
        self.S = S
        self.q = q
        self.r = r
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

    def get_stages_count(self) -> int:
        """
        Get number of stages/horizon length
        """
        return len(self.Q)
    
    def get_states_count(self, stage_id: int) -> int:
        """
        Get number of state variables in a given stage
        """
        return self.Q[stage_id].shape[0]
    
    def get_controls_count(self, stage_id: int) -> int:
        """
        Get number of control variables in a given stage
        """
        return self.R[stage_id].shape[0]


    def to_sparse(self) -> QP:
        """
        Transforms the OCP QP problem into QP problem with sparse formulation.
        """
        H_blocks = []
        for Qn, Rn, Sn in zip(self.Q, self.R, self.S):
            H_blocks.append(np.block([
                [Qn, Sn],
                [Sn, Rn],
            ]))
        H_rows = []
        H_offset = 0
        nV = sum(x.shape[0] for x in H_blocks)
        for H_block in H_blocks:
            size = H_block.shape[0] # H is square
            H_rows.append(np.concatenate([np.zeros((H_offset,size)), H_block, np.zeros((nV-H_offset-size, size))]))
            H_offset += size
        H = np.concatenate(H_rows, 0)
        C = np.zeros((0, nV))
        for i, (An, Bn) in enumerate(zip(self.A, self.B)):
            nX = self.get_states_count(i+1)
            C = np.append(C, np.concatenate([An, Bn, -np.identity(nX)]))
        d = np.concatenate([-x for x in self.b])
        q = np.array(zip(self.q, self.r)).flatten()
        # TODO
        return QP(H, q, None, None, None, None, None, C, d)
    
    def merge_stages(self, id: int) -> "QPOCP":
        """
        Creates new QPOCP with stages id and id+1 merged into one
        """
        A = self.A[id]
        B = self.B[id]
        b = self.b[id]
        Q0 = self.Q[id]
        R0 = self.R[id]
        S0 = self.S[id]
        Q1 = self.Q[id+1]
        R1 = self.R[id+1]
        S1 = self.S[id+1]
        Q_new = Q0 + A.dot(Q1).dot(A)
        R_new = np.block([
            [R0 + B.dot(Q1).dot(B), B.dot(S1)],
            [S1.T.dot(B), R1],
        ])
        S_new = np.concatenate([S0 + A.dot(Q1).dot(B), A.dot(S1)])
        Q_full = self.Q[:id] + [Q_new] + self.Q[id+2:]
        R_full = self.R[:id] + [R_new] +  self.R[id+2:]
        S_full = self.S[:id] + [S_new] + self.S[id+2:]
        if id + 1 < self.get_stages_count():
            A_new = ...
            B_new = np.concatenate([np.zeros(self.get_controls_count(id)), self.B[id+1]])
            b_new = self.b[id+1]
            A_full = self.A[:id] + [A_new] + self.A[id+2:]
            B_full = self.B[:id] + [B_new] + self.B[id+2:]
            b_full = self.b[:id] + [b_new] + self.b[id+2:]
        else:
            A_full = self.A[:id]
            B_full = self.B[:id]
            b_full = self.b[:id]
        lbu_new = np.concatenate(self.lbu[id:id+2])
        lbu_full = self.lbu[:id] + [lbu_new] + self.lbu[id+2:]
        ubu_new = np.concatenate(self.ubu[id:id+2])
        ubu_full = self.lbu[:id] + [ubu_new] + self.lbu[id+2:]
        lbx_full = self.lbx[:id+1] + self.lbx[id+1:]
        ubx_full = self.ubx[:id+1] + self.ubx[id+1:]
        #return QPOCP(Q_full, R_full, S_full, ..., ..., A_full, B_full, b_full, lbu_full, ubu_full, lbx_full, ubx_full)
        raise NotImplementedError("TODO")

    def to_dense(self) -> QP:
        """
        Transforms the OCP QP problem into QP problem with dense formulation.
        """
        return self.condense(1).to_sparse()

    def condense(self, N: int) -> "QPOCP":
        """
        Transforms the OCP QP problem into new OCP QP problem, which has N stages.
        """
        new_qp = self
        stages = self.get_stages_count()
        assert 0 < N <= stages
        min_merge = stages // N - 1
        stages_with_extra_merge = stages % N
        for new_stage_id in range(N):
            has_extra_marge = stages_with_extra_merge > 0
            merges = min_merge
            if has_extra_marge:
                stages_with_extra_merge -= 1
                merges += 1
            for _ in range(merges):
                new_qp = new_qp.merge_stages(new_stage_id)
        return new_qp