import numpy

class QPDense:
    """
    Class for standard form QP problems
    """
    def __init__(self, H: numpy.ndarray, q: numpy.ndarray):
        assert (H == H.T).all()
        self.H = H
        self.q = q

    def evaluate(self, x: numpy.ndarray):
        """
        Compute objective value
        """
        return x.T.dot(self.H).dot(x) + self.q.T.dot(x)