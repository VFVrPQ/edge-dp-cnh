from networkx.classes.function import edges
import numpy as np

def adjacencyMatrix2E(matrix):
    n, _ = matrix.shape

    E = list()
    for i in range(n):
        for j in range(i):
            if (matrix[i][j] == 1):
                E.append([i, j])
    return np.array(E)


def addSymmetricalEdges(E):
    '''传入np.arrray，表示无向图的下三角形边，恢复至2倍的边。保证传入的E没有u=v的边，且所有的边都是u>v的
        目前已知用的地方是在deepwalk加边之前。
    '''
    newE = list()
    D = dict()
    for i in range(len(E)):
        u, v = E[i][0], E[i][1]
        if u == v: continue
        # 正向的
        if not ((u, v) in D):
            D[(u, v)] = 1
            newE.append([u, v])
        # 反向的
        if not ((v, u) in D):
            D[(v, u)] = 1
            newE.append([v, u])
    return newE

    
def edges2AdjacencyMatrix(G):
    '''边转化为对称邻接矩阵。
    '''
    V, E = G
    n = len(V)

    M = np.zeros(shape=(n, n), dtype=int)
    for i in range(len(E)):
        u, v = E[i][0], E[i][1]
        M[u][v] = M[v][u] = 1

    return M

def calcDegree(E, n):
    '''计算每个节点的度（degree），返回np.array
        传进来的是无向图，且仅有一个方向的边（即有1-4， 就没有4-1的边）
    '''
    degreeArray = np.zeros(shape=(n,), dtype=int)
    for i in range(len(E)):
        u, v = E[i][0], E[i][1]
        if u == v: continue
        degreeArray[u] += 1
        degreeArray[v] += 1
    return degreeArray


def his2seq(histogram):
    '''histogram to sequence
    '''
    seq = []
    for i in range(len(histogram)):
        seq.extend([i] * histogram[i])
    # print(histogram, seq)
    return seq
