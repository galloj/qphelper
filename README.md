# QP helper

This Python package offers various utility functions for manipulating and verifying QP (Quadratic Programming) problems.

Install by:

```
pip install qphelper
```

This package contains two classes `QP` and `QPOCP`.

# Class `QP`

```py
from qphelper import QP
qp = QP(H, q, A, lbA, ubA, lb, ub, C, d)

# get primal objective value
print(qp.evaluate_primal(primal_solution))

# transform equalities into inequalities for solvers that do not support ehm
qp = qp.to_without_equalities()
```

# Class `OCPQP`

```py
from qphelper import QPOCP
qp = QPOCP(Q, R, S, ...)

# remove state variables from first stage by utilizing equality bounds
qp = qp.to_without_x0()

# transform the problem into dense `QP` object
qp_dense = qp.to_dense()
```
