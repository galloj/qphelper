from .qp import QP
import numpy as np
from typing import Optional

class QPOCP:
    r"""
    Class for OCP QP problems
    
    \[
    \min \sum_{t}^{N} \frac{1}{2}x^T_tQ_tx_t + \frac{1}{2}u^T_tR_tu_t + x^T_tS_tu_t + q_t^Tx_t + r_t^Tu_t
    \]

    \[
    x_{t+1} = A_{t}x_{t} + B_{t}u_{t} + b_{t}
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
    def __init__(self, Q: list[np.ndarray], R: list[np.ndarray], S: Optional[list[np.ndarray]], q: Optional[list[np.ndarray]], r: Optional[list[np.ndarray]], A: list[np.ndarray], B: list[np.ndarray], b: Optional[list[np.ndarray]], lbu: list[np.ndarray], ubu: list[np.ndarray], lbx: list[np.ndarray], ubx: list[np.ndarray], C: Optional[list[np.ndarray]], D: Optional[list[np.ndarray]], lg: Optional[list[np.ndarray]], ug: Optional[list[np.ndarray]]):
        N = len(Q)
        assert len(Q) == N
        assert len(R) == N
        if S is None:
            S = [np.zeros((Rn.shape[0], Qn.shape[0])) for Qn, Rn in zip(Q, R)]
        assert len(S) == N
        if q is None:
            q = [np.zeros(Qn.shape[0]) for Qn in Q]
        assert len(q) == N
        if r is None:
            r = [np.zeros(Rn.shape[0]) for Rn in R]
        assert len(r) == N
        assert len(A) == N-1
        assert len(B) == N-1
        if b is None:
            b = [np.zeros(Qn.shape[0]) for Qn in Q[1:]]
        assert len(b) == N-1
        assert len(lbu) == N
        assert len(ubu) == N
        assert len(lbx) == N
        assert len(ubx) == N
        assert (C is None) == (D is None) == (lg is None) == (ug is None)
        if C is None or D is None or lg is None or ug is None:
            C = [np.zeros((0, Qn.shape[0])) for Qn in Q]
            D = [np.zeros((0, Rn.shape[0])) for Rn in R]
            lg = [np.zeros(0)] * N
            ug = [np.zeros(0)] * N
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
            assert (lbut <= ubut).all()
            assert lbxt.shape == (nX,)
            assert ubxt.shape == (nX,)
            assert (lbxt <= ubxt).all()
            assert Ct.shape == (nC, nX)
            assert Dt.shape == (nC, nU)
            assert lgt.shape == (nC,)
            assert ugt.shape == (nC,)
            assert (lgt <= ugt).all()
        for t, (At, Bt, bt) in enumerate(zip(A, B, b)):
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

    def get_next_state(self, x: np.ndarray, u: np.ndarray, stage: int = 0) -> np.ndarray:
        r"""
        Calculates next state from the current state and control variables.

        \[
        x_{t+1} = A_{t}x_{t} + B_{t}u_{t} + b_{t}
        \]

        Parameters:
            x: state variables input (\(x_t\))
            u: control variables input (\(u_t\))
            stage: stage at which the variables have the given values (\(t\))

        Returns:
            The new state variables. (\(x_{t+1}\))
        """
        return self.A[stage].dot(x) + self.B[stage].dot(u) + self.b[stage]

    def to_sparse(self) -> QP:
        """
        Transforms the OCP QP problem into QP problem with sparse formulation.

        Returns:
            The new QP.
        """
        H_blocks = []
        for Qn, Rn, Sn in zip(self.Q, self.R, self.S):
            H_blocks.append(np.block([
                [Qn, Sn.T],
                [Sn, Rn],
            ]))
        H_rows = []
        H_offset = 0
        nV = sum(x.shape[0] for x in H_blocks)
        for H_block in H_blocks:
            size = H_block.shape[0] # H is square
            H_rows.append(np.concatenate([np.zeros((H_offset,size)), H_block, np.zeros((nV-H_offset-size, size))]))
            H_offset += size
        H = np.concatenate(H_rows, 1)
        C = np.zeros((0, nV))
        C_offset = 0
        for i, (An, Bn) in enumerate(zip(self.A, self.B)):
            nX = self.get_states_count(i+1)
            C = np.append(C, np.concatenate([np.zeros((An.shape[0], C_offset)), An, Bn, -np.identity(nX), np.zeros((An.shape[0], nV-C_offset-An.shape[1]-Bn.shape[1]-nX))], 1), 0)
            C_offset += self.get_states_count(i) +  self.get_controls_count(i)
        if len(self.b) == 0:
            d = np.zeros(0)
        else:
            d = np.concatenate([-x for x in self.b])
        q_arr = []
        for qn, rn in zip(self.q, self.r):
            q_arr += list(qn)
            q_arr += list(rn)
        q = np.array(q_arr)
        A = np.zeros((0, nV))
        A_offset = 0
        for i, (Cn, Dn) in enumerate(zip(self.C, self.D)):
            A = np.append(A, np.concatenate([np.zeros((Cn.shape[0], A_offset)), Cn, Dn, np.zeros((Cn.shape[0], nV-A_offset-Cn.shape[1]-Dn.shape[1]))], 1), 0)
            A_offset += Cn.shape[1] + Dn.shape[1]
        lbA = np.concatenate(self.lg)
        ubA = np.concatenate(self.ug)
        lb_arr = []
        for lbxn, lbun in zip(self.lbx, self.lbu):
            lb_arr += list(lbxn)
            lb_arr += list(lbun)
        ub_arr = []
        for ubxn, ubun in zip(self.ubx, self.ubu):
            ub_arr += list(ubxn)
            ub_arr += list(ubun)
        lb = np.array(lb_arr)
        ub = np.array(ub_arr)
        return QP(H, q, A, lbA, ubA, lb, ub, C, d)
    
    def merge_stages(self, id: int) -> "QPOCP":
        """
        Creates new QPOCP with stages id and id+1 merged into one

        Returns:
            The new OCP QP with merged stages.
        """
        assert 0 <= id < self.get_stages_count() - 1
        A = self.A[id]
        B = self.B[id]
        b = self.b[id]
        Q0 = self.Q[id]
        R0 = self.R[id]
        S0 = self.S[id]
        Q1 = self.Q[id+1]
        R1 = self.R[id+1]
        S1 = self.S[id+1]
        Q_add = A.T.dot(Q1).dot(A)
        assert np.allclose(Q_add, Q_add.T) # sanity check
        Q_add = (Q_add + Q_add.T) * 0.5 # fix symmetry rounding error issues
        Q_new = Q0 + Q_add
        R0_add = B.T.dot(Q1).dot(B)
        assert np.allclose(R0_add, R0_add.T) # sanity check
        R0_add = (R0_add + R0_add.T) * 0.5 # fix symmetry rounding error issues
        R_new = np.block([
            [R0 + R0_add, S1.dot(B).T],
            [S1.dot(B), R1],
        ])
        S_new = np.concatenate([S0 + A.T.dot(Q1).dot(B), S1.dot(A)])
        Q_full = self.Q[:id] + [Q_new] + self.Q[id+2:]
        R_full = self.R[:id] + [R_new] +  self.R[id+2:]
        S_full = self.S[:id] + [S_new] + self.S[id+2:]
        if id + 2 < self.get_stages_count():
            A1 = self.A[id+1]
            B1 = self.B[id+1]
            A_new = A1.dot(A)
            B_new = np.concatenate([A1.dot(B), B1], 1)
            b_new = A1.dot(b) + self.b[id+1]
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
        ubu_full = self.ubu[:id] + [ubu_new] + self.ubu[id+2:]
        lbx_full = self.lbx[:id+1] + self.lbx[id+2:]
        ubx_full = self.ubx[:id+1] + self.ubx[id+2:]
        # self.lbx[id+1] - b_t <= A_tx_t + B_tu_t <= self.ubx[id+1] - b_t
        # -> is added by C and D matrices
        q_new = self.q[id] + self.q[id+1].T.dot(A).T
        q_full = self.q[:id] + [q_new] + self.q[id+2:]
        r_new = np.concatenate([self.r[id] + self.q[id+1].T.dot(B).T, self.r[id+1] + S1.dot(b)])
        r_full = self.r[:id] + [r_new] + self.r[id+2:]
        bx_mask = (self.lbx[id+1] != -np.inf) | (self.ubx[id+1] != np.inf)
        C_new = np.concatenate([self.C[id], self.C[id+1].dot(A), A[bx_mask]], 0)
        C_full = self.C[:id] + [C_new] + self.C[id+2:]
        D_new = np.block([
            [self.D[id], np.zeros((self.D[id].shape[0], self.D[id+1].shape[1]))],
            [self.C[id+1].dot(B), self.D[id+1]],
            [B[bx_mask], np.zeros((B.shape[0], self.D[id+1].shape[1]))[bx_mask]]
        ])
        D_full = self.D[:id] + [D_new] + self.D[id+2:]
        lg_new = np.concatenate([self.lg[id], self.lg[id+1] - self.C[id+1].dot(b), (self.lbx[id+1] - b)[bx_mask]])
        lg_full = self.lg[:id] + [lg_new] + self.lg[id+2:]
        ug_new = np.concatenate([self.ug[id], self.ug[id+1] - self.C[id+1].dot(b), (self.ubx[id+1] - b)[bx_mask]])
        ug_full = self.ug[:id] + [ug_new] + self.ug[id+2:]
        return QPOCP(Q_full, R_full, S_full, q_full, r_full, A_full, B_full, b_full, lbu_full, ubu_full, lbx_full, ubx_full, C_full, D_full, lg_full, ug_full)

    def to_dense(self) -> QP:
        """
        Transforms the OCP QP problem into QP problem with dense formulation.

        Returns:
            The new dense QP.
        """
        return self.condense(1).to_sparse()

    def condense(self, N: int) -> "QPOCP":
        r"""
        Transforms the OCP QP problem into new OCP QP problem, which has N stages.

        The new optimization variables correspond to:
        \( [ x_0 u_0 u_1 ... u_t ] \)

        Returns:
            The condensed OCP QP.
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