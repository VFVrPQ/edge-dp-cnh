
import numpy as np
from math import fabs
from operator import pos
from numpy.core.fromnumeric import clip
from sklearn import metrics, svm
from scipy import spatial
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('../../')
from features.DP.postProcessing import PostProcessing


#=====================
class DistributionMetrics:
    def __init__():
        pass

    @staticmethod
    def normalize(a):
        ca = np.clip(a, a_min=0, a_max=None)
        na = ca / max(sum(ca), 1)
        return na

    @staticmethod
    def KSTest(pdf1, pdf2):
        cdf1 = PostProcessing.pdf2Cdf(pdf1)
        cdf2 = PostProcessing.pdf2Cdf(pdf2)
        n = len(cdf1)
        # ret = 0
        # for i in range(n):
        #     ret = max(ret, abs(cdf1[i]-cdf2[i]))
        ret = max(abs(np.array(cdf1)-np.array(cdf2)))
        # print('argmax', np.argmax(abs(np.array(cdf1)-np.array(cdf2))))
        return ret


    @staticmethod
    def normalizedKSTest(pdf1, pdf2):
        '''输入后，小于0的变成0，然后整体除以sum，做normalize。
        '''
        norm_pdf1 = DistributionMetrics.normalize(pdf1)
        norm_pdf2 = DistributionMetrics.normalize(pdf2)
        return DistributionMetrics.KSTest(norm_pdf1, norm_pdf2)

    @staticmethod
    def EMD(pdf1, pdf2):
        cdf1 = PostProcessing.pdf2Cdf(pdf1)
        cdf2 = PostProcessing.pdf2Cdf(pdf2)
        n = len(cdf1)
        ret = 0
        for i in range(n):
            ret += abs(cdf1[i] - cdf2[i])
        return ret
    
    @staticmethod
    def normalizedEMD(pdf1, pdf2):
        '''输入后，小于0的变成0，然后整体除以sum，做normalize。
        '''
        norm_pdf1 = DistributionMetrics.normalize(pdf1)
        norm_pdf2 = DistributionMetrics.normalize(pdf2)
        return DistributionMetrics.EMD(norm_pdf1, norm_pdf2)

    @staticmethod
    def aggr(pdf, T):
        '''前0~T-1的值聚合到T-1位置
        '''
        temp_pdf = np.array(pdf).copy()
        if T >= 1:
            temp_pdf[T-1] = np.sum(temp_pdf[0:T])
            temp_pdf[0:T-1] = 0
        return temp_pdf

    @staticmethod
    def normalizedEMD_T(pdf1, pdf2, T):
        '''输入后，小于0的变成0，然后整体除以sum，做normalize。
             distribution的前T-1个值挪动到第T-1个位置中, 前T-2个位置变成0, 其余位置不变
        '''
        temp_pdf1 = DistributionMetrics.aggr(pdf1, T)
        temp_pdf2 = DistributionMetrics.aggr(pdf2, T)

        norm_pdf1 = DistributionMetrics.normalize(temp_pdf1)
        norm_pdf2 = DistributionMetrics.normalize(temp_pdf2)
        return DistributionMetrics.EMD(norm_pdf1, norm_pdf2)

    @staticmethod
    def normalizedEMD_partial(pdf1, pdf2, n):
        '''输入后，小于0的变成0，然后整体除以sum，做normalize。仅计算前n项。
        '''
        norm_pdf1 = DistributionMetrics.normalize(pdf1)
        norm_pdf2 = DistributionMetrics.normalize(pdf2)
        return DistributionMetrics.EMD(norm_pdf1[:n], norm_pdf2[:n])


    @staticmethod
    def normalizedL1(pdf1, pdf2):
        '''输入后, 小于0的变成0, 然后整体除以sum, 做normalize。
            test:
                [0, 1, 1, 0, 1, 1, 1], 
                [0, 0, 0, 3, 0, 1, 1]
                1.2
        '''
        norm_pdf1 = DistributionMetrics.normalize(pdf1)
        norm_pdf2 = DistributionMetrics.normalize(pdf2)
        return np.linalg.norm(norm_pdf1-norm_pdf2, ord=1)
        


    @staticmethod
    def topKL1(hist1, hist2, kList):
        '''从hist中找到前k大的, 一一比较并返回。【前10, 前20, 前50】
            kList = [10, 20, 50, 0.01%, 0.05%] 注意0.01%,0.05%转换为个数。输入的k保证递增


            testok:
                    hist1 = [0, 1, 2, 2, 3, 4, 5]
                    hist2 = [0, 0, 0, 3, 3, 4, 5]
                    print(DistributionMetrics.topKL1(hist1, hist2, kList=[1, 2, 5, 10, 15]))
        '''
        errors = []

        for k in kList:
            s1 = []
            for i in range(len(hist1)-1, -1, -1):
                if hist1[i] > 0:
                    if len(s1) + hist1[i] >= k:
                        s1.extend([i]*int(k-len(s1)))
                        break
                    else:
                        s1.extend([i]*int(hist1[i]))
            s1 = np.asarray(s1)

            s2 = []
            for i in range(len(hist2)-1, -1, -1):
                if hist2[i] > 0:
                    if len(s2) + hist2[i] >= k:
                        s2.extend([i]*int(k-len(s2)))
                        break
                    else:
                        s2.extend([i]*int(hist2[i]))
            s2 = np.asarray(s2)

            l1 = np.linalg.norm(x=s1-s2, ord=1)
            errors.append(l1)
            # print('s1, s2, l1 =', s1, s2, l1)
        return np.asarray(errors)


    @staticmethod
    def bottomKEMD(hist1, hist2, kList):
        '''从hist中找到前k小的, 一一比较并返回。【前10, 前20, 前50】
            kList = [10, 20, 50, 0.01%, 0.05%] 注意0.01%,0.05%转换为个数。输入的k保证递增
        '''
        errors = []

        for k in kList:
            h1 = []
            for i in range(0, min(k, len(hist1))): # 从前往后，min时为了防止超过最大长度
                h1.append(hist1[i])
            h1 = np.asarray(h1)

            h2 = []
            for i in range(0, min(k, len(hist2))): # 从前往后
                h2.append(hist2[i])
            h2 = np.asarray(h2)

            emd = DistributionMetrics.EMD(h1, h2)
            errors.append(emd)
            # print(s1, s2, l1)
        return np.asarray(errors)




def auc(y, scores, pos_label=1):
    '''
        y: True binary labels.
        scores: 注意pos_label的score大
        pos_label: 正例的标签
    '''
    fpr, tpr, threshold = metrics.roc_curve(y_true=y, y_score=scores, pos_label=pos_label)
    # print('auc y =', y, 'scores =', scores)
    # print('auc fpr =', fpr, 'tpr =', tpr, 'threshold =', threshold)
    return metrics.auc(fpr, tpr)


def correlationDistance(u, v):
    '''correlation distance
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.correlation.html
    '''
    return spatial.distance.correlation(u=u, v=v)

def correlationDistance4Embedding(Z):
    '''计算n个embedding两两之间的correlationDistance
        Z: n*k
    '''
    # 中心化
    n, k = Z.shape
    curZ = Z.copy()
    for i in range(n):
        mu = np.average(curZ[i])
        curZ[i] = curZ[i] - mu
    
    ZZ = np.zeros(shape=(n, ))
    for i in range(n):
        zz = np.average(np.square(curZ[i]))
        ZZ[i] = np.sqrt(zz) # L2-norm

    
    A = np.zeros(shape=(n, n))
    for i in tqdm(range(n)):
        for j in range(i+1, n):
            uv = np.average(curZ[i]*curZ[j])
            if (ZZ[i] * ZZ[j] < 1e-8):
                if (uv < 1e-8): A[i][j] = 0
                else: A[i][j] = 1
            else:
                A[i][j] = np.abs(1 - uv / (ZZ[i] * ZZ[j])) # 防止负数出现
            
            # if i % 100 == 0 and j % 100 == 0: print('correlationDistance {}, {}'.format(i, j))
    return A

def sigmoid(z):
    return 1/(1 + np.exp(-z))

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


from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

def accuracyNodeClassification(X, y):
    '''node classfication的准确率。
        X是embedding，y是labels
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    clf = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return metrics.accuracy_score(y_test, y_pred)


import networkx as nx
import random
from sklearn.linear_model import LogisticRegression

def constructDataset(V, E):
    '''「类似[word2vec, dpne]」：为了获得正样本，我们随机移除50%的边且保证剩余网络的连通性；
    而负样本是从没有边的点对间随机抽样相同数量的边获得。

    具体做法：1.先求MST，保证连通性；
    2. 除MST外的边随机删除50% * |E|
    返回
    # 3. 再从完全图的边中剩余的边抽样
        V: np.array
        E: np.array
       posEdges, fakeEdges, delEdges: return, 样本和对应标签；缺失的边和对应的标签
    '''
    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)

    # 1.MST
    T = nx.minimum_spanning_tree(G)
    print('T.edges', len(T.edges))
    
    # 2.得到可以删除的边集，作为测试集
    nV, nE = len(V), len(E)
    delEdges = random.sample(G.edges - T.edges, int(0.5 * nE)) # 得到要预测的值

    # 3.得到负样本
    CG = nx.complete_graph(nV)
    fakeEdges = random.sample(CG.edges - G.edges, int(0.5 * nE))

    # 4.得到正样本[注意这边会删除部分边]
    G.remove_edges_from(delEdges)
    posEdges = list(G.edges) # 正样本 EdgeView和list不能加法

    return posEdges, [1]*len(posEdges), fakeEdges, [0]*len(fakeEdges), delEdges, [1]*len(delEdges)
    

def expandEdgesAndLabels(x, y, emb_vertex, emb_context, op):
    '''将边展开成embeddings
    '''
    emb_vertex = np.asarray(emb_vertex)
    emb_context = np.asarray(emb_context)
    
    s_idx, t_idx, y_train = [], [], []
    for (s, t), yy in zip(x, y):
        s_idx.append(s)
        t_idx.append(t)
        y_train.append(yy)
    s_train = emb_vertex[s_idx]
    if emb_context is not None:
        t_train = emb_context[t_idx]
    else:
        t_train = emb_vertex[t_idx]
    #
    x_train = op(s_train, t_train)
    y_train = np.asarray(y_train)
    return x_train, y_train


def hadamard(x, y):
    return x*y

def accuracyLinkPrediction(x, y, delx, dely, emb_vertex, emb_context, op=hadamard):
    '''link prediction返回准确率。
    具体做法：在link prediction中，我们会扔掉50%的边，在剩余图上得到node embedding 再去预测缺失的边 
        X: edges
        y: labels
        op: operator
    '''
    print('begin[accuracyLinkPrediction].............')
    # 2. 将边展开成embeddings
    x_train, y_train = expandEdgesAndLabels(x=x, y=y, emb_vertex=emb_vertex, emb_context=emb_context, op=op)

    # 3. 训练
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(x_train, y_train)
    
    # 4. predict
    # 使用缺失的边作为节点和标签，预测
    x_test, y_test = expandEdgesAndLabels(x=delx, y=dely, emb_vertex=emb_vertex, emb_context=emb_context, op=op)
    accuracy = clf.score(x_test, y_test)
    print('complete[accuracyLinkPrediction].............')
    return accuracy

import sys
sys.path.append('../../')
from features.common.matrix import flattenMatrix, convertTo01comp, flattenMatrix2    
def accuracyUnsurpervisedLinkPrediction(Z, m, oA):
    '''非监督学习的link prediction，只考虑原图有边的precision。
    准确率不好估计了。
    具体做法：根据发布的embedding，做$S[z_{u}, z_{v}] = z_{u} \cdot z_{v}$，组成相似矩阵$S$。取上三角前m大的边当作新图，与原图比较看准确率（即隐私性）。
        Z: n*k, embedding matrix
        m: number of edges
        oA: origin graph，邻接矩阵的表示
        op: operator
    '''
    print('begin[accuracyUnsurpervisedLinkPrediction].............')
    n, k = Z.shape
    A = np.dot(Z, Z.T)
    d1A = flattenMatrix(M=A) # 1维且只有上三角的A数组

    # partition是从小到大排列的，因此要倒着取
    # 将# >= 第m大小值的设为1
    pos = n * (n - 1) // 2 - m + 1
    mthValue = np.partition(a=d1A, kth=pos)[pos-1]
    # print(A, pos, mthValue)
    A01 = np.where(A >= mthValue, 1, 0)
    fA = flattenMatrix(M=A01)
    foA = flattenMatrix(M=oA)
    # 【只判断总体的准确率显然是不行的，还要单独看有边的准确率，这个还需要学习一下】
    # 目前只考虑有边的准确率
    cnt, all = 0, sum(foA)
    for i in range(len(foA)):
        if (foA[i] == 1) and (fA[i] == 1):
            cnt += 1
    accuracy = cnt / all
    print('end[accuracyUnsurpervisedLinkPrediction]............. acc = ', accuracy)
    return accuracy

# def accuracyUnsurpervisedLinkPrediction2(Z, oA):
#     '''非监督学习的link prediction，只考虑原图有边的准确率。
#     准确率不好估计了。
#     具体做法：根据发布的embedding，做$S[z_{u}, z_{v}] = z_{u} \cdot z_{v}$，组成相似矩阵$S$。做sigmoid再取round，与原图比较看准确率（即隐私性）。
#         Z: n*k, embedding matrix
#         oA: origin graph，邻接矩阵的表示
#     '''
#     print('begin[accuracyUnsurpervisedLinkPrediction2].............')
#     A = np.round(sigmoid(np.dot(Z, Z.T)))
#     # print('Z', list(Z))
#     print('A', list(np.dot(Z, Z.T)), A) # 点积的结果太近了，且A全是1
#     fA = flattenMatrix(M=A) # 1维且只有上三角的A数组
#     foA = flattenMatrix(M=oA)
#     # 【只判断总体的准确率显然是不行的，还要单独看有边的准确率，这个还需要学习一下】
#     # 目前只考虑有边的准确率
#     cnt, all = 0, sum(foA)
#     for i in range(len(foA)):
#         if (foA[i] == 1) and (fA[i] == 1):
#             cnt += 1
#     accuracy = cnt / all
#     print('end[accuracyUnsurpervisedLinkPrediction2].............')
#     return accuracy

from sklearn.cluster import KMeans
def accuracyUnsurpervisedLinkPrediction3(Z, oA):
    '''非监督学习的link prediction，考虑auc, 准确率和召回率。
    具体做法：https://arxiv.org/pdf/2005.02131.pdf attack0
        Z: n*k, embedding matrix
        oA: origin graph，邻接矩阵的表示
    '''
    print('begin[accuracyUnsurpervisedLinkPrediction].............')
    A = correlationDistance4Embedding(Z=Z)
    d1A = flattenMatrix2(M=A) # 1维且只有上三角的A数组


    print('begin[aucUnsurpervisedLinkPrediction].............')
    scores = 1 - flattenMatrix(M=A) # score大的是pos_label
    trueLabels = flattenMatrix(M=oA) # 1维且只有上三角的A数组
    aucLP = auc(y=trueLabels, scores=scores, pos_label=1) # sklearn.metrics计算auc
    print('end[aucUnsurpervisedLinkPrediction].............auc =', aucLP)

    # 使用kmeans计算推测的值
    print('np.histogram(flattenMatrix(A))', np.histogram(flattenMatrix(A), bins=2))
    print('np.histogram(flattenMatrix(A))', np.histogram(flattenMatrix(A), bins=3))
    print('np.histogram(flattenMatrix(A))', np.histogram(flattenMatrix(A), bins=4))
    kmeans = KMeans(n_clusters=2, random_state=0).fit(d1A)
    # print('Z =', Z, 'd1A =', d1A, 'kmeans.cluster_centers_', kmeans.cluster_centers_)
    n, k = Z.shape
    # The cluster with lower (higher) average distance value is considered as the set of positive (negative) node pairs.
    predA = np.where(A < (kmeans.cluster_centers_[0][0] + kmeans.cluster_centers_[1][0]) / 2., 1, 0)
    # for i in range(n):
    #     for j in range(i+1, n):
    #         '''The cluster with lower (higher) average distance value is considered as the set of positive (negative) node pairs.'''
    #         positive = 1
    #         if fabs(A[i][j] - kmeans.cluster_centers_[0][0]) < fabs(A[i][j] - kmeans.cluster_centers_[1][0]):
    #             if kmeans.cluster_centers_[0][0] > kmeans.cluster_centers_[1][0]:
    #                 positive = 0
    #         else: 
    #             if kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0]:
    #                 positive = 0
    #         A[i][j] = positive
            
    fA = flattenMatrix(M=predA)
    foA = flattenMatrix(M=oA)
    ### 【只判断总体的准确率显然是不行的，还要单独看有边的准确率，这个还需要学习一下】
    # 这个是排序问题的精确率和召回率
    cnt = 0
    pall, rall = sum(fA), sum(foA)
    for i in range(len(foA)):
        if (foA[i] == 1) and (fA[i] == 1):
            cnt += 1
    
    if pall == 0: precision = 0 
    else: precision = cnt / pall

    if rall == 0: recall = 0
    else: recall = cnt / rall
    print('end[accuracyUnsurpervisedLinkPrediction].............')
    print('precision, recall, pall, rall', precision, recall, pall, rall)
    return precision, recall, aucLP, pall, rall


def f1ScoreUnsurpervisedLinkPrediction3(Z, oA):
    '''非监督学习的link prediction，考虑f1Score【均衡点】。
    具体做法：计算correlationDistance后，计算f1Score
        Z: n*k, embedding matrix
        oA: origin graph，邻接矩阵的表示
    '''
    print('begin[f1ScoreUnsurpervisedLinkPrediction3].............')

    trueLabels = flattenMatrix(M=oA) # 1维且只有上三角的A数组
    m = sum(trueLabels) # 总共的边数88234

    A = correlationDistance4Embedding(Z=Z)
    d1A = flattenMatrix(M=A) # 1维且只有上三角的A数组

    # 对前m个值预测
    # partition是从小到大排列的，将# <= 第m大小值的设为1
    mthValue = np.partition(a=d1A, kth=m)[m-1]
    A01 = np.where(A <= mthValue, 1, 0)
    yPred = flattenMatrix(M=A01) # score大的是pos_label

    f1Score = metrics.f1_score(y_true=trueLabels, y_pred=yPred, pos_label=1, average="binary") # sklearn.metrics计算f1_score
    # 辅助测试，看是否相同
    precScore = metrics.precision_score(y_true=trueLabels, y_pred=yPred, pos_label=1, average="binary")
    recallScore = metrics.recall_score(y_true=trueLabels, y_pred=yPred, pos_label=1, average="binary")
    print('end[f1ScoreUnsurpervisedLinkPrediction3].............mthValue =', mthValue, 
        ', f1Score =', f1Score, ', precScore =', precScore, ', recallScore =', recallScore)
    return f1Score

def approxError(A, B, ord=2):
    '''sum_{i}||R_{i}||/ sum_{i}||A_{i}||
        A: n*n, 邻接矩阵
        B: n*k, one-hot矩阵，节点属于哪个类
    '''
    R = A - A @ B @ B.T
    rA = np.sum(a=np.linalg.norm(x=A, ord=ord, axis=1))
    rR = np.sum(a=np.linalg.norm(x=R, ord=ord, axis=1))
    return rR / rA


if __name__ == '__main__':





    cdf1 = [0, 1, 2, 2, 3, 4, 5]
    cdf2 = [0, 0, 0, 3, 3, 4, 5]
    print(DistributionMetrics.aggr(cdf1, 3))

    print(DistributionMetrics.topKL1(cdf1, cdf2, kList=[1, 2, 5, 10, 15])) # cdf权当pdf
    print(DistributionMetrics.normalizedL1(PostProcessing.cdf2Pdf(cdf1), PostProcessing.cdf2Pdf(cdf2)))
    print(DistributionMetrics.EMD(cdf1, cdf2), DistributionMetrics.KSTest(cdf1, cdf2))
    print('pdf1, pdf2=', PostProcessing.cdf2Pdf(cdf1), PostProcessing.cdf2Pdf(cdf2))
    print('normalized', DistributionMetrics.normalizedEMD(PostProcessing.cdf2Pdf(cdf1), PostProcessing.cdf2Pdf(cdf2)),
            DistributionMetrics.normalizedKSTest(PostProcessing.cdf2Pdf(cdf1), PostProcessing.cdf2Pdf(cdf2)))
    



    # 测试的时候把n_splits改成=2
    # edges = np.array([[1, 2], [0, 1]])
    # y = np.array([1, 0])
    # emb_vertex = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # acc = accuracyLinkPrediction(edges=edges, y=y, emb_vertex=emb_vertex, emb_context=None, op=hadamard)
    # print(acc)

    # 测试accuracyUnsurpervisedLinkPrediction
    Z = np.array([[1, 2], [2, -2], [3, 4], [5, 6]])
    m = 2
    oA = np.array([[0, 1, 0, 0],
                   [1, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])
    # acc = accuracyUnsurpervisedLinkPrediction2(Z=Z, oA=oA)
    # print(acc, (6-m) / 6)
    


    y = np.array([1, 1, 2, 2])
    scores = np.array([0.1, 0.4, 0.35, 0.8])
    print(auc(y=y, scores=scores, pos_label=2))


