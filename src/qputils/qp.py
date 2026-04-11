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
        result += (ybA * self.ubA)[(self.ubA != -numpy.inf) & (ybA > 0)].sum()
        result += (-yb * self.lb)[(self.lb != -numpy.inf) & (yb < 0)].sum()
        result += (yb * self.ub)[(self.ub != -numpy.inf) & (yb > 0)].sum()
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
        raise NotImplementedError("TODO")

    def calculate_dual(self, x: numpy.ndarray):
        """
        Calculate dual

        Parameters:
            x: nV - primal solution
        """
        raise NotImplementedError("TODO")

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

    def to_identity_hessian(self) -> "QP":
        """
        Converts the problem to one, which has identity Hessian.
        Requires that the Hessian is positive definite.
        """
        qp = self.to_without_bounds()
        raise NotImplementedError("TODO")

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