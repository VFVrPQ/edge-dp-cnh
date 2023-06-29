"""
2012-ICDT-Differentially Private Summaries for Sparse Data

one-side filter-sample
"""
import numpy as np
from diffprivlib.mechanisms import Geometric

class GeometricFilterAndSample:
    r"""Geometric，然后filter，再sample

    Parameters
    ----------
    epsilon : float
        Privacy paramter

    sensitivity : float, 
        The sensitivity of the mechanism.  Must be in [0, ∞).
    
    theta: int, default 0 表示没有限制
        Filter's threshold, >=保留，<则取thetaInitial

    thetaInitial: int

    tau: int, default 1 表示没有限制
        Sampling's threhold, 以`p_{i} = min(v_{i} / tau, 1)`概率保留
    """
    def __init__(self, epsilon, sensitivity, theta=0, tau=1, thetaInitial=0):
        self._geometric = Geometric(epsilon=epsilon, sensitivity=sensitivity)
        self.theta = theta
        self.thetaInitial = thetaInitial
        self.tau = tau

    def randomise(self, M):
        """Randomise `M` with the mechanism

        Parameters
        ----------
        M : np.array
            The matrix to be randomised.

        Returns
        -------
        noisyM : np.array
            The randomised matrix.
        """
        noisyM = M.copy()
        n, k = M.shape

        for i in range(n):
            for j in range(k):
                noisyM[i][j] = self._geometric.randomise(M[i][j])
                # print(i, j, M[i][j], noisyM[i][j], end=' ')
                # filter
                if noisyM[i][j] < self.theta: noisyM[i][j] = self.thetaInitial
                # print(noisyM[i][j], end=' ')
                # sample
                if min(1. * noisyM[i][j] / self.tau, 1.) < np.random.random(): noisyM[i][j] = self.thetaInitial
                # print(noisyM[i][j])
        return noisyM


if __name__ == '__main__':
    gfas = GeometricFilterAndSample(epsilon=1, sensitivity=1, theta=4, tau=10)
    a = np.random.randint(0, 15, size=(6, 3))
    # print(a)
    noisy_a = gfas.randomise(M=a)