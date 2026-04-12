import numpy
from typing import Optional

class QP:
    r"""
    Class for standard form QP problems.

    The problems have following form:

    \[
    \min \frac{1}{2} x^THx + qx^T
    \]

    \[
    \begin{equation}
    \begin{split}
    lbA & \leq & Ax & \leq & ubA \\
    lb  & \leq & x  & \leq & ub \\
    & & Cx & = & d
    \end{split}
    \end{equation}
    \]

    Parameters:
        H: nV x nV matrix
        q: nV array
        A: nC x nV matrix
        lbA: nC array
        ubA: nC array
        lb: nV array
        ub: nV array
        C: nD x nV matrix
        d: nD array
    """
    def __init__(self, H: numpy.ndarray, q: numpy.ndarray, A: Optional[numpy.ndarray] = None, lbA: Optional[numpy.ndarray] = None, ubA: Optional[numpy.ndarray] = None, lb: Optional[numpy.ndarray] = None, ub: Optional[numpy.ndarray] = None, C: Optional[numpy.ndarray] = None, d: Optional[numpy.ndarray] = None):
        assert (H == H.T).all()
        nV = H.shape[0]
        self.H = H
        self.q = q
        if A is not None:
            assert A.shape[1] == nV
            self.A = A
        else:
            self.A = numpy.zeros((nV, 0))
        nC = self.A.shape[0]
        if lbA is not None:
            assert lbA.shape == (nC,)
            self.lbA = lbA
        else:
            self.lbA = numpy.full((nC,), -numpy.inf)
        if ubA is not None:
            assert ubA.shape == (nC,)
            self.ubA = ubA
        else:
            self.ubA = numpy.full((nC,), numpy.inf)
        if lb is not None:
            assert lb.shape == (nV,)
            self.lb = lb
        else:
            self.lb = numpy.full((nV,), -numpy.inf)
        if ub is not None:
            assert ub.shape == (nV,)
            self.ub = ub
        else:
            self.ub = numpy.full((nV,), numpy.inf)
        assert (C is None) == (d is None)
        if C is not None and d is not None:
            self.C = C
            self.d = d
        else:
            self.C = numpy.zeros((nV, 0))
            self.d = numpy.zeros((0,))

    def get_equalities_count(self) -> int:
        return self.d.shape[0]
    
    def get_inequalities_count(self) -> int:
        return self.lbA.shape[0]
    
    def get_variables_count(self) -> int:
        return self.H.shape[0]

    def evaluate_primal(self, x: numpy.ndarray) -> float:
        """
        Compute objective value

        Parameters:
            x: nV - primal solution
        """
        return  float(0.5 * (x.T.dot(self.H).dot(x)) + self.q.T.dot(x))
    
    def evaluate_dual(self, x: numpy.ndarray, ybA: Optional[numpy.ndarray] = None, yb: Optional[numpy.ndarray] = None, yD: Optional[numpy.ndarray] = None) -> float:
        """
        Compute objective value for dual problem

        Parameters:
            ybA: nC - dual solution for general constraints
            yb: nV - dual solution for variable bounds
            yD: nD - dual solution for equalities
        """
        if ybA is None:
            ybA = numpy.zeros((self.get_inequalities_count(),))
        if yb is None:
            yb = numpy.zeros((self.get_variables_count(),))
        if yD is None:
            yD = numpy.zeros((self.get_equalities_count(),))
        result = 0.5 * (x.T.dot(self.H).dot(x))
        result += (-ybA * self.lbA)[(self.lbA != -numpy.inf) & (ybA < 0)].sum()
        result += (ybA * self.ubA)[(self.ubA != numpy.inf) & (ybA > 0)].sum()
        result += (-yb * self.lb)[(self.lb != -numpy.inf) & (yb < 0)].sum()
        result += (yb * self.ub)[(self.ub != numpy.inf) & (yb > 0)].sum()
        result += (yD * self.d).sum()
        return -float(result)
    
    def calculate_primal(self, ybA: Optional[numpy.ndarray] = None, yb: Optional[numpy.ndarray] = None, yD: Optional[numpy.ndarray] = None):
        """
        Calculate primal

        Parameters:
            ybA: nC - dual solution for general constraints
            yb: nV - dual solution for variable bounds
            yD: nD - dual solution for equalities
        """
        if ybA is None:
            ybA = numpy.zeros((self.get_inequalities_count(),))
        if yb is None:
            yb = numpy.zeros((self.get_variables_count(),))
        if yD is None:
            yD = numpy.zeros((self.get_equalities_count(),))
        A = numpy.concatenate([self.H, self.C])
        b = numpy.concatenate([-(self.A.T.dot(ybA) + yb + self.C.T.dot(yD) + self.q), self.d])
        eps = 0.0001
        for ybAn, An, lbAn, ubAn in zip(ybA, self.A, self.lbA, self.ubA):
            if abs(ybAn) > eps:
                A = numpy.append(A, numpy.array([An]), 0)
                if ybAn < 0:
                    b = numpy.append(b, [lbAn])
                else:
                    b = numpy.append(b, [ubAn])
        for i, (ybn, lbn, ubn) in enumerate(zip(yb, self.lb, self.ub)):
            if abs(ybn) > eps:
                new_constr = numpy.zeros(self.get_variables_count())
                new_constr[i] = 1.0
                A = numpy.append(A, numpy.array([new_constr]), 0)
                if ybn < 0:
                    b = numpy.append(b, [lbn])
                else:
                    b = numpy.append(b, [ubn])
        primal, _, _, _ = numpy.linalg.lstsq(A, b)
        return primal

    def calculate_dual(self, x: numpy.ndarray):
        """
        Calculate dual

        Parameters:
            x: nV - primal solution
        """
        A = numpy.concatenate([self.A.T, numpy.identity(self.get_variables_count()), self.C.T], 1)
        b = -(self.H.dot(x) + self.q)
        dual_length = self.get_inequalities_count() + self.get_variables_count() + self.get_equalities_count()
        eps = 0.0001
        for i, (An, lbn, ubn) in enumerate(zip(self.A, self.lbA, self.ubA)):
            if not(An.T.dot(x) <= lbn + eps or An.T.dot(x) >= ubn - eps):
                new_constr = numpy.zeros(dual_length)
                new_constr[i] = 1.0
                A = numpy.append(A, numpy.array([new_constr]), 0)
                b = numpy.append(b, numpy.zeros(1))
        for i, (xn, lbn, ubn) in enumerate(zip(x, self.lb, self.ub)):
            if not(xn <= lbn + eps or xn >= ubn - eps):
                new_constr = numpy.zeros(dual_length)
                new_constr[self.get_inequalities_count()+i] = 1.0
                A = numpy.append(A, numpy.array([new_constr]), 0)
                b = numpy.append(b, numpy.zeros(1))
        dual, _, _, _ = numpy.linalg.lstsq(A, b)
        offset = 0
        ybA = dual[offset:offset+self.get_inequalities_count()]
        offset += self.get_inequalities_count()
        yb = dual[offset:offset+self.get_variables_count()]
        offset += self.get_variables_count()
        yD = dual[offset:]
        return (ybA, yb, yD)

    def calculate_primal_residual(self, x: numpy.ndarray)-> numpy.ndarray:
        result = 0
        result = numpy.absolute(self.C.dot(x) - self.d).max(initial=result)
        result =  (self.A.dot(x) - self.ubA).max(initial=result)
        result =  (self.lbA - self.A.dot(x)).max(initial=result)
        result =  (x - self.ub).max(initial=result)
        result =  (self.lb - x).max(initial=result)
        return result

    def calculate_dual_residual(self, x: numpy.ndarray, ybA: Optional[numpy.ndarray] = None, yb: Optional[numpy.ndarray] = None, yD: Optional[numpy.ndarray] = None) -> float:
        if ybA is None:
            ybA = numpy.zeros((self.get_inequalities_count(),))
        if yb is None:
            yb = numpy.zeros((self.get_variables_count(),))
        if yD is None:
            yD = numpy.zeros((self.get_equalities_count(),))
        return numpy.absolute(self.H.dot(x) + self.A.T.dot(ybA) + self.C.T.dot(yD) + yb + self.q).max(initial = 0)

    def calculate_duality_gap(self, x: numpy.ndarray, ybA: Optional[numpy.ndarray] = None, yb: Optional[numpy.ndarray] = None, yD: Optional[numpy.ndarray] = None) -> float:
        return self.evaluate_primal(x) - self.evaluate_dual(x, ybA, yb, yD)

    def to_identity_hessian(self) -> tuple["QP", numpy.ndarray]:
        """
        Converts the problem to one, which has identity Hessian.
        Requires that the Hessian is positive definite.
        """
        qp = self.to_without_bounds()
        H_tr = numpy.linalg.cholesky(qp.H)
        H_tr_inv = numpy.linalg.inv(H_tr)
        # v^T = x^T H_tr
        # x^T = v^T H_tr_inv
        # x = H_tr_inv^T v
        q_new = qp.q.dot(H_tr_inv)
        A_new = qp.A.dot(H_tr_inv.T)
        C_new = qp.C.dot(H_tr_inv.T)
        return (QP(numpy.identity(qp.get_variables_count()), q_new, A_new, qp.lbA, qp.ubA, None, None, C_new, qp.d), H_tr_inv)

    def to_without_bounds(self) -> "QP":
        """
        Converts QP into QP with variable bounds transformed into general constraints.
        This transformation does not alter primal objective function.
        """
        newA = self.A
        newlbA = self.lbA
        newubA = self.ubA
        for i, (lbn, ubn) in enumerate(zip(self.lb, self.ub)):
            is_set = lbn != -numpy.inf or ubn != numpy.inf
            if not is_set:
                continue
            newlbA = numpy.append(newlbA, lbn)
            newubA = numpy.append(newubA, ubn)
            new_constaint = numpy.concatenate([numpy.zeros(i), numpy.ones(1), numpy.zeros(self.get_variables_count()-i-1)])
            newA = numpy.concatenate([newA, numpy.array([new_constaint])], 0)
        return QP(self.H, self.q, newA, newlbA, newubA, None, None, self.C, self.d)
    
    def to_without_general_lower_bounds(self) -> "QP":
        """
        Converts QP into QP with lower general bounds transformed into upper general constraints.
        This transformation does not alter primal objective function.
        """
        newA = self.A
        newubA = self.ubA
        for lbAn, An in zip(self.lbA, self.A):
            if lbAn == -numpy.inf:
                continue
            newubA = numpy.append(newubA, -lbAn)
            newA = numpy.concatenate([newA, numpy.array([-An])], 0)
        return QP(self.H, self.q, newA, None, newubA, self.lb, self.ub, self.C, self.d)

    def to_without_general_constraints(self) -> "QP":
        """
        Converts the problem to one, which has empty A matrix.
        """
        raise NotImplementedError("TODO")