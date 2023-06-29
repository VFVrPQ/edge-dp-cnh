import numpy as np
import numpy as np

def normalization_prob(matrix, eps=1e-6):
    sum = np.sum(a=matrix, axis=1).reshape(-1, 1)
    sum[ np.abs(sum - 0) < eps ] = 1.
    
    norm_matrix = matrix / sum
    return norm_matrix

def normalization_col(matrix, r=5):
    '''按列normalization，先变成[0, 1]，再变成[-1/2, 1/2]，最后变成 [-r, r]
    '''
    mx, mn = np.max(a=matrix, axis=0), np.min(a=matrix, axis=0)
    for i in range(len(mx)):
        if (mx[i] == mn[i]):
            print('i=%d: mx[i]=mn[i]=%d' % (i, mx[i]))
            mx[i] = mn[i] + 1
            # return np.zeros_like(matrix)
    norm_matrix = (matrix-mn) / (mx - mn)
    norm_matrix = (norm_matrix - 0.5) * 2 * r
    return norm_matrix

def normalization_col_nonnegative(matrix, r=5):
    '''按列normalization，先变成[0, 1]，最后变成 [0, r]
    '''
    mx, mn = np.max(a=matrix, axis=0), np.min(a=matrix, axis=0)
    for i in range(len(mx)):
        if (mx[i] == mn[i]):
            print('i=%d: mx[i]=mn[i]=%d' % (i, mx[i]))
            mx[i] = mn[i] + 1
            # return np.zeros_like(matrix)
    norm_matrix = (matrix - mn) / (mx - mn)
    norm_matrix = norm_matrix * r
    return norm_matrix

def test_normalization_col():
    matrix = np.array([[1, 2], [3, 4], [5, 6]])

    nm = normalization_col(matrix=matrix)
    print(nm)


def test_normalization_prob():
    matrix = np.array([[1, 2], [3, 4], [5, 6]])
    nm = normalization_prob(matrix)
    print(nm)


if __name__ == '__main__':
    # test_normalization_col()
    test_normalization_prob()

