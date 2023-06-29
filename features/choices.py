'''budget allocation和consistency constraints
'''
import numpy as np
from sklearn import metrics


def best_budget_allocation(n, k, epsilon):
    '''最佳隐私分配$\epsilon_{2} / \epsilon_{1}$的比例 ✔
    '''
    r = ((k + 1) / 8. / n) ** (1. / 3) # epsilon_2 / epsilon_1
    epsilon_1 = 1 / (1 + r) * epsilon
    epsilon_2 = r / (1 + r) * epsilon
    return epsilon_1, epsilon_2


def best_consistency_constraints(p_node_cluster, p_cluster_cluster, node_belong, ac, an):
    '''
        node_belong: , 是一个公开的信息。第一步是随机的，所以没什么信息暴露；以后的隐私性都是有post-processing保证的。
        ac: float, $ac$ is the cofficient for p_cluster_cluster[c][j], and $ac$ is for p_cluster_cluster[c][c];
        an: float, $an$ is the cofficient for p_node_cluster[i][c]

        returns optimal_node_cluster, optimal_cluster_cluster:
    '''
    ## 0. preparation
    n, k = p_node_cluster.shape

    # 0.A. node_number_per_cluster[c]表示第c个cluster有多少个nodes
    node_number_per_cluster = np.zeros(k, dtype=np.int64) 
    for i in range(n):
        node_number_per_cluster[ node_belong[i] ] += 1

    # 0.B. 表示第i个cluster对第j个cluster的perturbed_node_degree之和
    node_degree_per_cluster = np.zeros((k, k), dtype=np.float64)
    for i in range(n): # node
        for c in range(k): # cluster
            node_degree_per_cluster[ node_belong[i] ][ c ] += p_node_cluster[ i ][ c ]

    ## 1. Bottom-up traversal
    # 1.A. p_cluster_cluster[c][c]
    optimal_cluster_cluster = np.zeros((k, k), dtype=np.float64)
    for c in range(k):
        if node_number_per_cluster[c] == 0:
            optimal_cluster_cluster[c][c] = 0
        else:
            optimal_cluster_cluster[c][c] = (ac * p_cluster_cluster[c][c] + (2 * an / node_number_per_cluster[c]) * node_degree_per_cluster[c][c]) / (ac + (4 * an / node_number_per_cluster[c]))

    # 1.B. p_cluster_cluster[c][j] (j < c)
    for c in range(k):
        for j in range(c): # j < c
            if (node_number_per_cluster[c] == 0) or (node_number_per_cluster[j] == 0):
                optimal_cluster_cluster[c][j] = 0
            else:
                nc = node_number_per_cluster[c]
                nj = node_number_per_cluster[j]
                optimal_cluster_cluster[c][j] = (ac * p_cluster_cluster[c][j] + (an / nc) * node_degree_per_cluster[c][j] + (an / nj) * node_degree_per_cluster[j][c]) / (ac + an / nc + an / nj)

    ## 2. Top-down traversal
    optimal_node_cluster = np.zeros((n, k), dtype=np.float64)
    for i in range(n): # node
        for c in range(k): # cluster
            if (node_number_per_cluster[ node_belong[i] ] == 0) or (node_number_per_cluster[ c ] == 0): ## 前一项必定不为0，hhh
                optimal_node_cluster[i][c] = 0 # 从直观上来说，就是0；且与optimal_cluster_cluster保持一致
            else:
                if node_belong[i] == c:
                    oc = 2 * optimal_cluster_cluster[ node_belong[i] ][ c ]
                elif node_belong[i] > c:
                    oc = optimal_cluster_cluster[ node_belong[i] ][ c ]
                elif node_belong[i] < c: # 只有矩阵的下三角才有值
                    oc = optimal_cluster_cluster[ c ][ node_belong[i] ]
                optimal_node_cluster[i][c] = p_node_cluster[i][c] + 1.0 / node_number_per_cluster[ node_belong[i] ] * ( oc - node_degree_per_cluster[ node_belong[i] ][c])

    return optimal_node_cluster, optimal_cluster_cluster


def end_condition(last_node_belong, node_belong):
    '''
        returns : float, 相同的个数占总node数的比例。
    '''
    return metrics.adjusted_mutual_info_score(last_node_belong, node_belong)


def calc_AMI(last_node_belong, node_belong):
    '''计算AMI
        returns : float, 越大越好。
    '''
    return metrics.adjusted_mutual_info_score(last_node_belong, node_belong)

def calc_ARI(A, B):
    '''计算ARI
    '''
    return metrics.adjusted_rand_score(A, B)

def calcAMI_ARI(A, B):
    return calc_AMI(A, B), calc_ARI(A, B)

def unattributed_consistency(S):
    '''2010-Boosting the Accuracy of Differentially Private Queries Through Consistency
        S: 序列

        时间复杂度O(n^3)
        这里用不了，复杂度太高了。
    '''
    n = len(S)
    qsum = np.zeros(shape=(n,), dtype=np.float64)
    qsum[0] = S[0] 
    for i in range(1, n): qsum[i] = qsum[i-1] + S[i]

    M = np.zeros(shape=(n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i, n):
            if i==0:
                M[i][j] = qsum[j] / (j-i+1)
            else:
                M[i][j] = (qsum[j] - qsum[i-1]) / (j-i+1)

    new_S = np.zeros_like(S, dtype=np.float64)
    for k in range(n):
        new_S[k] = float('inf')
        for j in range(k, n):
            t = M[j][j]
            for i in range(j):
                t = max(t, M[i][j])
            new_S[k] = min(new_S[k], t)
    return new_S

def test_best_budget_allocation():
    n=1005
    k=42
    epsilon=2
    print(best_budget_allocation(n, k, epsilon))



def best_consistency_constraints_test_simple1():
    node_cluster = np.array([[1, 1],
        [1, 0],
        [1, 2],
        [0, 2],
        [0, 2]], dtype=np.float64)
    cluster_cluster = np.array([[1, 1],
        [1, 3]], dtype=np.float64)
    node_belong = [0, 0, 1, 1, 1]
    n, k = node_cluster.shape

    from degree import laplace_nk
    epsilon = 1.
    epsilon1, epsilon2 = best_budget_allocation(n=n, k=k, epsilon=epsilon)
    p_node_cluster = laplace_nk(matrix=node_cluster, mu=0, b=2./epsilon1)
    p_cluster_cluster = laplace_nk(matrix=cluster_cluster, mu=0, b=1./epsilon2)

    ac = epsilon1 ** 2 / 8.
    an = epsilon2 ** 2 / 2.

    o_nc, o_cc = best_consistency_constraints(p_node_cluster=p_node_cluster,
        p_cluster_cluster=p_cluster_cluster,
        node_belong=node_belong,
        ac=ac,
        an=an)
    print('o_nc =', o_nc)
    print('o_cc =', o_cc)
    
    ms = np.zeros(shape=(k, k))
    for i in range(n):
        for j in range(k):
            ms[ node_belong[i] ][j] += o_nc[i][j]
    print('ms =', ms)


def test_unattributed_consistency():
    #S = np.array([14, 9, 10, 15])
    S = np.array([1, 2, 0, 11])
    print(unattributed_consistency(S))


if __name__ == '__main__':
    best_consistency_constraints_test_simple1()
    test_best_budget_allocation()
    test_unattributed_consistency()


    A = [0, 0, 1, 2]
    B = [0, 0, 1, 1]
    print(calc_ARI(A, B))