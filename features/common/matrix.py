import numpy as np


def convertTo01(matrix):
    '''将矩阵内的元素都变成最近的值，0或1
    '''
    m = matrix.copy()
    m[ m < 0.5] = 0
    m[ m >= 0.5] = 1
    return m

def convertTo01comp(matrix, v):
    '''根据v的大小判断等于1还是等于0
    '''
    m = matrix.copy()
    m[ matrix < v] = 0
    m[ matrix >= v] = 1 # 注意第二步一定要用matrix，如果用m因为值已经变过了，所以可能会导致错误
    return m

def toRowProb(matrix, eps=1e-8):
    '''将矩阵按行转化成概率矩阵。
    '''
    s = np.sum(a=matrix, axis=1).reshape(-1, 1) # 一行一行加
    s[ np.fabs(s) < eps ] = 1. ### 1e-8不一定正确，是随便设置的一个较小值
    probMatrix = matrix / s
    return probMatrix


def sumM1ToMn(M, n):
    '''返回矩阵(M+M^{2}+···+M^{n})/n。规定M矩阵是方阵
        M: matrix
        n: number
    '''
    row, col = M.shape
    assert row == col, 'sumM1ToMn：矩阵M不是方阵'
    Mk, ans = np.eye(row), np.zeros_like(M)
    for i in range(n):
        Mk = np.dot(Mk, M)
        ans = ans + Mk
    ans = ans / n
    return ans

def flattenMatrix(M):
    '''将上三角摊平成一维
    '''
    n, m = M.shape
    assert n==m, "flattenMatrix: n!=m"
    
    res = list()
    for i in range(n):
        for j in range(i+1, m):
            res.append(M[i][j])
    return np.array(res)

def flattenMatrix2(M):
    '''将上三角摊平成一维list
    '''
    n, m = M.shape
    assert n==m, "flattenMatrix: n!=m"
    
    res = list()
    for i in range(n):
        for j in range(i+1, m):
            res.append([M[i][j]])
    return np.array(res)

if __name__ == '__main__':
    m = np.random.rand(4, 3)
    print(m)
    print(convertTo01(m))
    print(m / 2)


    n = np.random.rand(4, 3)
    print('n =', n)
    print('n + m =', n + m)
    
    p = np.array([[1, 2], [3, 4]])
    q = np.array([[1], [2]])
    print('p * q =', np.dot(p, q))


    print('(p + p^2)/2 =', sumM1ToMn(M=p, n=2))

    # sumM1ToMn(M=q, n=2)


    print('log(p) =', np.log(p))