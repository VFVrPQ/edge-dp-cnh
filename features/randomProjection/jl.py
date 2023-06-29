'''
https://github.com/hrishikeshv/Johnson-LindenstraussLemma/blob/master/jl.py
'''
import numpy as np
import math

def jl(data, k=None, method='Gaussian', sigma=None):
    '''Johnson-Lindenstrauss Lemma
        :params data: (n, d), 
        :params k: int, assume k > 4 / (epsilon^2 - epsilon^3) ln(2*n^2)  %k > 8 ln(2n^2) / (epsilon^{2})
        :params sigma: float, 随机矩阵的方差。
    '''
    n, d = data.shape
    if k == None:
        epsilon = 0.25 ## 
        k = int(math.ceil(4 * math.log(2*n*n) / (epsilon ** 2 - epsilon ** 3)))
    if method == 'Gaussian':
        if sigma == None:
            scale = 1. / np.sqrt(k)
        else:
            scale = sigma
        print('d, k=', d, k)
        P = np.random.normal(loc=0., scale=scale, size=(d, k))
    return data @ P

def edgedp_jl(data, epsilon, delta, k):
    '''(epsilon, delta)-edge DP
        :params data: (n, n)
        :params epsilon: float, 
        :params delta: float, > 0，暂时考虑取
        :params k: int, projected dimension

        :params Y: n*k
    '''
    
    n, _ = data.shape

    old_old_sigma = 4 * np.sqrt(2) / epsilon * np.sqrt(np.log(1./delta))
    old_sigma = 2 * np.sqrt(2) / epsilon * np.sqrt(2*(np.log(1./ delta) + epsilon))

    x = np.log(n) + np.log(2 / delta)
    sigma = ( np.sqrt(2) + np.sqrt(4 * x / k) ) / epsilon * np.sqrt(2*(np.log(1./ delta) + epsilon))
    print('k=%d, sigma=%f, old_sigma=%f' % (k, sigma, old_sigma))

    Y = jl(data=data, k=k, method='Gaussian')
    Z = Y + np.random.normal(loc=0., scale=sigma, size=(n, k))
    return Z


def directJLAddNoise(data, ratio, k):
    '''direct JL的方法
        :params data: (n, n)
        :params ratio: sigma_{n}^{2} / sigma^{2}
        :params k: int, projected dimension

        :params Y: n*k
    '''
    n, _ = data.shape

    sigma = 1. / np.sqrt(k)
    Y = jl(data=data, k=k, method='Gaussian', sigma=sigma)
    sigma_n = sigma * np.sqrt(ratio) ## sigma_{n}^{2} = sigma^{2} * ratio
    Z = Y + np.random.normal(loc=0., scale=sigma_n, size=(n, k))
    return Z