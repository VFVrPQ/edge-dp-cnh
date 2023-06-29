'''
从features/degree.py得到
'''
import numpy as np

def laplace_nk(matrix, mu=0, b=1):
    '''输出matrix中每个元素独立同分布地加 均值为mu，scale为b的Laplace噪声
        b: sens / epsilon
        returns p_matrix：numpy.ndarray
    '''
    # n, k = matrix.shape
    p_matrix = matrix + np.random.laplace(loc=mu, scale=b, size=matrix.shape)#(n, k))
    return p_matrix
    

def laplace(value, mu=0, b=1):
    '''value加 均值为mu，scale为b的Laplace噪声
        returns p_value: float
    '''
    return value + np.random.laplace(loc=mu, scale=b)


if __name__ == '__main__':
    print(type(laplace(1, mu=0, b=1))) # float