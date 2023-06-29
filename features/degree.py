### 度的计算，包括node cluster和cluster cluster，以及Laplace噪声的矩阵
from features.DP.GeometricFilterAndSample import GeometricFilterAndSample
import numpy as np
import random

import sys
sys.path.append('../')
from features.choices import best_consistency_constraints

def calc_node_node(E, n):
    '''
    '''
    n_E = len(E) # the number of edges
    
    node_node = np.zeros((n, n), dtype=np.int64)
    for e in range(n_E):
        u, v = E[e][0], E[e][1]
        node_node[u][v] += 1
        node_node[v][u] += 1
    return node_node

def calcAdjacencyMatrixDiagonal(E, n):
    '''计算邻接矩阵，且对角元素=1
    '''
    A = calc_node_node(E=E, n=n)
    for i in range(n): A[i][i] = 1
    return A

def calc_node_degree(E, node_belong, k):
    '''node_belong: [c1, c2]表示node所属的cluster
        returns: node_degree, n*k的矩阵，表示user i在cluster j的degree
    '''
    nE = len(E) # the number of edges
    n = len(node_belong)
    
    node_degree = np.zeros((n, k), dtype=np.int64)
    for i in range(nE):
        u, v = E[i][0], E[i][1]
        node_degree[u][ node_belong[v] ] += 1
        node_degree[v][ node_belong[u] ] += 1
    return node_degree

def laplace_nk(matrix, mu=0, b=1):
    '''输出matrix中每个元素独立同分布地加 均值为mu，scale为b的Laplace噪声
        returns p_matrix：numpy.ndarray
    '''
    n, k = matrix.shape
    p_matrix = matrix + np.random.laplace(loc=mu, scale=b, size=(n, k))
    return p_matrix

# Step 3:
def calc_cluster_degree(E, node_belong, k):
    '''node_belong: node_belong[i]=j表示node i所属的cluster j
        returns: cluster_degree, k*k的矩阵，表示cluster i在cluster j的degree
        ✔
    '''
    n_E = len(E) # the number of edges
    
    cluster_degree = np.zeros((k, k), dtype=np.int64)
    for i in range(n_E):
        u, v = E[i][0], E[i][1]
        if node_belong[u] < node_belong[v]: # 仅给node_belong[u] >= node_belong[v]的部分加（下三角）
            u, v = v, u
        
        cluster_degree[ node_belong[u] ][ node_belong[v] ] += 1
    return cluster_degree

def calc_cluster_degree_xz(cluster_degree, k=None):
    '''✔
    '''
    if k == None: # to-test
        k, _ = cluster_degree.shape

    cluster_degree_xz = np.zeros_like(cluster_degree, dtype=np.float64)
    for i in range(k):
        for j in range(k):
            if i == j:
                cluster_degree_xz[i][j] = cluster_degree[i][j] # 暂不处理，先保持不变
            elif i > j:
                cluster_degree_xz[i][j] = cluster_degree[i][j] # 不变
            else:
                cluster_degree_xz[i][j] = cluster_degree[j][i] # 对称的赋值过来
    return cluster_degree_xz


def calc_node_cluster_pp(node_cluster, node_belong, b=None):
    '''UNATTR_CONS: 表示unattributed constraints，将n*k的矩阵摊平，并记录编号
        b: 噪声绝对值的期望
    '''
    n, k = node_cluster.shape
    num_pc = np.zeros(shape=(k, ), dtype=np.float64) # num_pc[c]代表cluster c中有多少个节点。
    for i in range(n):
        num_pc[ node_belong[i] ] += 1

    nc_pp = node_cluster.copy()
    #print('id =', id(node_cluster), id(nc_pp))
    if b!=None: nc_pp = nc_pp - b
    nc_pp[ nc_pp < 0 ] = 0
    for i in range(n):
        for j in range(k):
            if nc_pp[i][j] > num_pc[j]: # node i连向cluster j的边不会超过cluster j内节点的个数（这个需要再考虑，可能有重边之类的）
                nc_pp[i][j] = num_pc[j]
    return nc_pp

def calc_cluster_cluster_pp(cluster_cluster, node_belong):
    '''没有重边
    '''
    n = len(node_belong)
    k = len(cluster_cluster)

    num_pc = np.zeros(shape=(k, ), dtype=np.float64) # num_pc[c]代表cluster c中有多少个节点。
    for i in range(n):
        num_pc[ node_belong[i] ] += 1
    
    cc_pp = cluster_cluster.copy()
    cc_pp[ cc_pp < 0 ] = 0
    for i in range(k):
        for j in range(k):
            if cc_pp[i][j] > num_pc[i] * num_pc[j]: # 这应该不容易达到
                cc_pp[i][j] = num_pc[i] * num_pc[j] 
    return cc_pp

def calc_NC(E, node_belong, k, used_epsilon, have_pp, have_noise):
    #, useLabel=False):
    '''计算node-cluster degree (dpgen调用)'''
    node_cluster = calc_node_degree(E, node_belong, k)
    if have_noise == True:
        p_node_cluster = laplace_nk(node_cluster, mu=0, b=2. / used_epsilon)
    else:
        print('have_noise=False')
        p_node_cluster = node_cluster

    if (have_pp == True) and (have_noise == True):
        s_node_cluster = calc_node_cluster_pp(node_cluster=p_node_cluster,
            node_belong=node_belong,
            b = 2./ used_epsilon)
    else:
        print('have_pp=False or have_noise=False')
        s_node_cluster = p_node_cluster

    return s_node_cluster

def calc_NC_CC(E, node_belong, k, used_epsilon1, used_epsilon2, have_cc=True, have_noise=True, have_pp=True):
        # useLabel=False
    '''计算node_cluster, cluster_cluster
        have_pp: 
    '''
    node_cluster = calc_node_degree(E, node_belong, k)
    cluster_cluster = calc_cluster_degree(E, node_belong, k)
    if have_noise == True:
        p_node_cluster = laplace_nk(node_cluster, mu=0, b=2. / used_epsilon1) # 平均分的隐私预算
        p_cluster_cluster = laplace_nk(matrix=cluster_cluster, mu=0, b=1. / used_epsilon2)
    else:
        print('have_noise=False')
        p_node_cluster = node_cluster
        p_cluster_cluster = cluster_cluster

    if (have_cc == True) and (have_noise == True):
        s_node_cluster, s_cluster_cluster = best_consistency_constraints(p_node_cluster=p_node_cluster, 
                        p_cluster_cluster=p_cluster_cluster, 
                        node_belong=node_belong,
                        ac=(used_epsilon1 ** 2) / 2.,
                        an=(used_epsilon2 ** 2) / 8.)
    else: # 用作对比consistency constraints的实验
        print('have_cc=False or have_noise=False')
        s_node_cluster, s_cluster_cluster = p_node_cluster, p_cluster_cluster

    s_cluster_cluster = calc_cluster_degree_xz(s_cluster_cluster, k=k)

    if have_pp==True:
        s_node_cluster_pp = calc_node_cluster_pp(node_cluster=s_node_cluster, node_belong=node_belong)#, b=2. / used_epsilon1)
        s_cluster_cluster_pp = calc_cluster_cluster_pp(cluster_cluster=s_cluster_cluster, node_belong=node_belong)
    else:
        print('calc_NC_CC have_pp=False')
        s_node_cluster_pp = s_node_cluster
        s_cluster_cluster_pp = s_cluster_cluster

    print('NC = {}, CC = {}'.format(node_cluster, cluster_cluster))
    print('s_NC = {}, s_CC = {}'.format(s_node_cluster_pp, s_cluster_cluster_pp))
    return s_node_cluster_pp, s_cluster_cluster_pp


def calc_NC_CC22(E, node_belong, k, used_epsilon, have_cc=True, have_noise=True, have_pp=True):
    '''计算node_cluster, cluster_cluster。通过node_cluster计算cluster_cluster
        have_pp: post-processing，最后的<0，收缩回0;
    '''
    node_cluster = calc_node_degree(E, node_belong, k)
    cluster_cluster = calc_cluster_degree(E, node_belong, k)

    n, _ = node_cluster.shape
    if have_noise == True:
        p_node_cluster = laplace_nk(node_cluster, mu=0, b=2. / used_epsilon) # 平均分的隐私预算
        p_node_cluster[ p_node_cluster < 0 ] = 0 # 负值变成0

        p_cluster_cluster = np.zeros(shape=(k, k), dtype=np.float64) 
        for u in range(n):
            for B in range(k):
                A = node_belong[u]
                p_cluster_cluster[A][B] += p_node_cluster[u][B] # 这里只要node_cluster[u][B] / cc[ belong[u] ][B]对u之和满足1即可
    else:
        print('have_noise=False')
        p_node_cluster = node_cluster
        p_cluster_cluster = cluster_cluster

    print('have_cc=False or have_noise=False')
    s_node_cluster, s_cluster_cluster = p_node_cluster, p_cluster_cluster

    s_cluster_cluster = calc_cluster_degree_xz(s_cluster_cluster, k=k)

    #####将负值变成0，虽然会破坏consistency constraints，但是结果更重要
    print('calc_NC_CC have_pp=False')
    s_node_cluster_pp = s_node_cluster
    s_cluster_cluster_pp = s_cluster_cluster
    
    return s_node_cluster_pp, s_cluster_cluster_pp


def calc_NC_CC_GFS(E, nodeLabels, k, epsilon,
        theta=0, tau=1, thetaInitial=0):
    '''geometric distribution & filter and sample
        计算node_cluster, cluster_cluster。通过node_cluster计算cluster_cluster
    '''
    node_cluster = calc_node_degree(E, nodeLabels, k)
    n, _ = node_cluster.shape

    gfas = GeometricFilterAndSample(epsilon=epsilon, sensitivity=2, 
        theta=theta, tau=tau, thetaInitial=thetaInitial)
    p_nc = gfas.randomise(M=node_cluster)

    # 计算cluster_cluster
    p_cc = np.zeros(shape=(k, k), dtype=np.float64) 
    for u in range(n):
        for B in range(k):
            A = nodeLabels[u]
            # to-do 这里A=B的时候会变成2倍
            p_cc[A][B] += p_nc[u][B] # 这里只要node_cluster[u][B] / cc[ belong[u] ][B]对u之和满足1即可

    return p_nc, p_cc



def NC2NN(node_cluster, node_belong, cluster_cluster=None):
    '''✔
    '''
    n, k = node_cluster.shape
    nc = node_cluster.copy()
    nc[ nc < 0 ] = 0 # 负值变成0
    
    row = np.sum(nc, axis=1).reshape(-1, 1)
    row[ np.fabs(row-0) < 1e-8 ] = 1
    prob_row_nc = nc / row

    if cluster_cluster == None:
        cc = np.zeros(shape=(k, k), dtype=np.float64) 
        for u in range(n):
            for B in range(k):
                A = node_belong[u]
                cc[A][B] += nc[u][B] 
    else:
        cc = cluster_cluster.copy()
        cc[ cc < 0 ] = 0
        for A in range(k): 
            cc[A][A] *= 2 # 对角线乘2
    cc[ np.fabs(cc-0) < 1e-8 ] = 1 # 是0，则变成1
    
    prob_nn = np.zeros(shape=(n, n), dtype=np.float64)
    for u in range(n):
        for v in range(n):
            A, B = node_belong[u], node_belong[v]
            if u == v: # 
                prob_nn[u][v] = 0
            elif A == B:
                if np.fabs(nc[v][A]) < 1e-8:
                    prob_nn[u][v] = 0.
                else:
                    prob_nn[u][v] = prob_row_nc[u][B] * (nc[v][A] / (cc[B][A] - nc[u][A]))
            else:
                prob_nn[u][v] = prob_row_nc[u][B] * (nc[v][A] / cc[B][A])
    return prob_nn


def add_least_edges(node_cluster, L):
    '''对于边数少于L的节点，添加到L条边。添加方式为Preferential Attachment，即按照每个cluster的边数概率选择
        要保证node_cluster的元素>=0
    '''
    num_edges_of_cluster = np.sum(node_cluster, axis=0) 
    num_edges_of_node = np.sum(node_cluster, axis=1)
    #print(num_edges_of_cluster, type(num_edges_of_cluster), num_edges_of_cluster.shape)
    n, k = node_cluster.shape

    cids = [i for i in range(k)]
    
    nodes = [i for i in range(n)]
    random.shuffle(nodes)

    count = 0
    nc = node_cluster.copy()
    for node in nodes:
        if num_edges_of_node[node] < L:
            count += 1
        while num_edges_of_node[node] < L:
            p = num_edges_of_cluster / sum(num_edges_of_cluster)
            c = np.random.choice(a=cids, p=p) # cluster
            #print(node, c)
            nc[node][c] += 1
            num_edges_of_node[node] += 1
            num_edges_of_cluster[c] += 1
    print('the number of nodes whose edges number is less than L=%d is %d' %(L, count))
    return nc
