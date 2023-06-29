import numpy as np

class ReportNoisyMax:
    '''2014Dwork book有三种实现
        - laplace
        - EM
        - exponential noise
    '''
    def __init__(self):
        pass

    @staticmethod
    def exponentialNoise(q, eps=1.0, sensitivity=1.0, prng=np.random, monotonic=False):
        '''2021-arXiv-THE PERMUTE-AND-FLIP MECHANISM IS IDENTICAL TO REPORT-NOISY-MAX WITH EXPONENTIAL NOISE
        '''
        coef = 1.0 if monotonic else 0.5

        v = q + prng.exponential(scale=sensitivity/(coef*eps), size=q.size)
        # print(np.random.exponential(scale=sensitivity/(coef*eps), size=q.size))
        return np.argmax(v)


if __name__ == '__main__':
    print(ReportNoisyMax.exponentialNoise(np.array([1, 2, 3]), eps=0.05))