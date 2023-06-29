import numpy as np
from numpy.core.einsumfunc import einsum_path

def calcD(alpha, ratio):
    '''
        alpha > 1
        ratio: sigma_{n}^{2} / sigma^{2}
    '''
    a = 1. / 2 * np.log( (3. + ratio) / (2. + ratio)) + 1. / 2 / (alpha - 1) * np.log((3. + ratio) / (alpha * (3. + ratio) - (alpha - 1) * (2 + ratio)))
    b = 1. / 2 * np.log( (2. + ratio) / (3. + ratio)) + 1. / 2 / (alpha - 1) * np.log((2. + ratio) / (alpha * (2. + ratio) - (alpha - 1) * (3 + ratio)))
    return 2. * max(a, b)

def calcd(D, epsilon):
    return int(np.floor(epsilon / D))

def calcED(epsilon, delta, alpha):
    return epsilon + np.log(1. / delta) / (alpha - 1), delta

def calcDirectJLParams(alpha, ratio, delta, epsilon):
    '''
        epsilon是RDP的epsilon，不是DP的
    '''
    D = calcD(alpha=alpha, ratio=ratio)
    d = calcd(D=D, epsilon=epsilon)
    assert D * d <= epsilon, "D * d > epsilon"
    print(' "D * d <= epsilon"', D*d, epsilon)
    e, _ = calcED(epsilon=epsilon, delta=delta, alpha=alpha)
    return d, e

if __name__ == '__main__':
    n = 4039
    delta = 1. / n / n / 10
    for ratio in [1, 2, 4, 8, 16]:
        for alpha in [2, 4, 8, 16]:
            if alpha < 3 + ratio:
                for epsilon in [1, 2, 3, 4, 5]:
                    D = calcD(alpha=alpha, ratio=ratio)
                    d = calcd(D=D, epsilon=epsilon)
                    e, _ = calcED(epsilon=epsilon, delta=delta, alpha=alpha)
                    if e < 7:
                        print('ration=%d, alpha=%d, epsilon=%d, D=%f, d=%d, e=%f' % (ratio, alpha, epsilon, D, d, e))
    # ration=8, alpha=8, epsilon=3, D=0.076686, d=39, e=5.701441
    # ration=16, alpha=8, epsilon=2, D=0.016287, d=122, e=4.701441
    # ration=16, alpha=8, epsilon=3, D=0.016287, d=184, e=5.701441