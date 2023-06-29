import numpy as np

class PermuteAndFlip:
    '''
        Permute-and-Flip: A new mechanism for differentially private selection
        github link: https://github.com/ryan112358/permute-and-flip
    '''
    def __init__(self):
        pass

    @staticmethod
    def pf(q, eps=1.0, sensitivity=1.0, prng=np.random, monotonic=False):
        '''
            q: quality
            from https://github.com/ryan112358/permute-and-flip/blob/main/mechanisms.py
        '''
        coef = 1.0 if monotonic else 0.5

        q = q - q.max()
        p = np.exp(coef*eps/sensitivity*q)
        # print('q =', q)
        # print('p =', p)

        for i in prng.permutation(p.size):
            # print(i, prng.rand(), p[i])
            if prng.rand() <= p[i]:
                return i


if __name__ == '__main__':
    print(PermuteAndFlip.pf(np.array([1, 2, 3])))