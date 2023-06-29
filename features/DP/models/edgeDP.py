'''edge DP

subgraphCounting
'''

from collections import defaultdict
from re import A
import time
import numpy as np
from tqdm import tqdm

class GraphStat(object):
    '''mainly store aggregated statistics of G
        Parameters
        ----------
        G: networkx graph

        Returns
        -------
        nodesNum: 节点个数

        maxA: 最大度数

        A: 任意点对间公共邻居的集合

        References
        ----------
        .. https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/util.py#L150

    '''
    def __init__(self, G):
        # degree number
        self.nodesNum = len(G.nodes())

        # A_ij: the set of common neighbors of i and j
        # a_ij: the number of common neighbors of i and j
        self.A, self.a = defaultdict(set), defaultdict(int)
        self.maxA = -1.0
        self.initSparseA(G)
        # print('Gstat: ', self.nodesNum, self.A, self.maxA)

    def initSparseA(self, G):
        startTime = time.time()
        for u, v in G.edges(): 
            for p in G[u]:
                if p != v:
                    self.A['{},{}'.format(min(v, p), max(v, p))].add(u)
            for p in G[v]:
                if p != u:
                    self.A['{},{}'.format(min(u, p), max(u, p))].add(v)
        print('---------initSparseA: {} seconds--------------'.format(time.time() - startTime))

        for ij, commonNeighbors in self.A.items():
            self.a[ij] = len(commonNeighbors)
            self.maxA = max(self.maxA, len(commonNeighbors))

    def getA(self, i, j):
        '''公共邻居集合
        '''
        return self.A['{},{}'.format(i, j)]

    def geta(self, i, j):
        '''公共邻居个数
        '''
        # return len(self.getA(i, j))
        return self.a['{},{}'.format(i, j)]


class GraphStatCommonNeighbors(object):
    '''mainly store aggregated statistics of G
        Parameters
        ----------
        G: networkx graph

        Returns
        -------
        nodesNum: 节点个数

        maxA: 最大度数

        A: 任意点对间公共邻居的集合

        References
        ----------
        .. https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/util.py#L150

    '''
    def __init__(self, G):
        # degree number
        self.nodesNum = len(G.nodes())

        # A_ij: the set of common neighbors of i and j
        # a_ij: the number of common neighbors of i and j
        self.a = np.zeros(shape=(self.nodesNum, self.nodesNum), dtype=int)
        # self.hist = [0 for i in range(self.nodesNum-1)] # 最大是n-2,hist统计
        self.hist = np.zeros(shape=(self.nodesNum-1,), dtype=np.int64)
        self.maxA = -1.0
        self.initSparseA(G)
        # print('Gstat: ', self.nodesNum, self.A, self.maxA)

    def initSparseA(self, G):
        startTime = time.time()

        self.hist[0] = np.int64(self.nodesNum * (self.nodesNum - 1) / 2)
        for u in range(len(G.nodes())):
            neList = [v for v in G[u]]
            # 邻居序列里的均是公共邻居
            for p in range(len(neList)):
                for q in range(p):
                    v, w = neList[p], neList[q]
                    if v>w: v, w = w, v
                    self.a[v][w] += 1
                    self.a[w][v] += 1
                    # 上述两个算同一个点对
                    self.hist[ self.a[v][w] ] += 1
                    self.hist[ self.a[v][w]-1 ] -= 1
            # print(neList)
        
            # for p in range(len(G[u])):
                # for q in range(p):


        # for u, v in G.edges(): 
        #     for p in G[u]:
        #         if p != v:
        #             self.A['{},{}'.format(min(v, p), max(v, p))].add(u)
        #     for p in G[v]:
        #         if p != u:
        #             self.A['{},{}'.format(min(u, p), max(u, p))].add(v)
        
        print('---------initSparseA: {} seconds--------------'.format(time.time() - startTime))
        self.maxA = np.max(self.a)
        # for ij, commonNeighbors in self.A.items():
        #     self.a[ij] = len(commonNeighbors)
        #     self.maxA = max(self.maxA, len(commonNeighbors))

    def geta(self, i, j):
        '''公共邻居个数
        '''
        return self.a[i][j]


class AliasMethod:
    '''给定一个离散型随机变量的概率分布规律 [公式] ，希望设计一个方法能够从该概率分布中进行采样使得采样结果尽可能服从概率分布 [公式]
        Alias方法将整个概率分布压成一个 1*N 的矩形，对于每个事件i ，转换为对应矩形中的面积为 N * p_i / (sum_{i}p_i)。
        通过上述操作，一般会有某些位置面积大于1某些位置的面积小于1。我们通过将面积大于1的事件多出的面积补充到面积小于1对应的事件中，以确保每一个小方格的面积为1，同时，保证每一方格至多存储两个事件。
        维护两个数组accept和alias,accept数组中的accept[i]表示事件i占第i列矩形的面积的比例。 alias[i]表示第i列中不是事件i的另一个事件的编号。
        在进行采样的时候，每次生成一个随机数 i \in [0, N) ,再生成一个随机数 r ~ uniform(0, 1) ，若 r < accept[i] ，则表示接受事件i，否则，拒绝事件 i 返回alias[i].
        预处理alias table的时间复杂度仍为 O(N) ,而每次采样产生事件的时间复杂度为 O(1) 。

        Parameters
        ----------
        truth: list, 长度为N，所有的值都是非负的

        Returns
        -------
        value: 返回下标，每次采样时间复杂度是O(1)

        References
        ----------
        .. https://zhuanlan.zhihu.com/p/54867139
    '''
    def __init__(self, truth):
        self.N = len(truth)
        area_ratio = list(np.array(truth) / np.sum(truth) * self.N) # 注意要除以np.sum(truth)，变成1*N的矩阵
        self.createAliasTable(area_ratio=area_ratio)

    def createAliasTable(self, area_ratio):
        '''创建alias table
            Parameters
            ----------
            area_ratio: list, 长度为N，和为N.

            Returns
            -------
            accept: accept数组中的accept[i]表示事件i占第i列矩形的面积的比例。

            alias: alias[i]表示第i列中不是事件i的另一个事件的编号。
        '''
        l = len(area_ratio)
        accept, alias = [0] * l, [0] * l
        small, large = [], []

        for i, prob in enumerate(area_ratio):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = area_ratio[small_idx]
            alias[small_idx] = large_idx
            area_ratio[large_idx] = area_ratio[large_idx] - (1 - area_ratio[small_idx])
            if area_ratio[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        self.accept = accept
        self.alias = alias
        # return accept, alias
    
    def sample(self):
        N = self.N
        i = int(np.random.random()*N)
        r = np.random.random()
        if r < self.accept[i]:
            return i
        else:
            return self.alias[i]
        

class EdgeDPLadderMechanism:
    r"""
    Parameters
    ----------
    epsilon : float,

    G: networkx graph

    References
    ----------
    .. https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/Algorithms/edgeDP_subgraphCounting.py#L80


    """
    def __init__(self, G):
        self.G = G
        self.Gstat = GraphStat(G)
        # print('G.nodes(), G.edges()', G.nodes(), G.edges() )

    def fit(self, epsilon, sens, queryType, trueCount=None, ladders=None):
        '''计算三角形个数、所有点对的公共邻居。
            sens: sensitivity
        '''
        G, Gstat = self.G, self.Gstat
        if trueCount == None: trueCount = self.count(G, Gstat, queryType)

        # ladders: ladder function evaluated on G
        if ladders == None: ladders = self.ladderFunction(G, Gstat, queryType)
        # M: length of the ladder function
        M = len(ladders)

        ranges = [0.]
        weights = [1.0] # the center's weight

        # rungs 1 to M
        dst = 0.0
        for t in range(M):
            weights.append(2*ladders[t]*np.exp(epsilon/2.0/sens*(-t-1)))
            dst = dst + ladders[t] # M+1个值，相邻两个值组成范围
            ranges.append(dst)
            
        # rung M+1
        weights.append(2*ladders[-1]*np.exp(epsilon/2.0/sens*(-M-1)) / (1-np.exp(-epsilon/2.0/sens)))
        # print('ranges, weights :', ranges[:10],  weights[:10])

        ####the only part that involves randomness, may store the earlier result for evaluation over multiple runs
        if queryType == 'triangle':
            noisyCount = self.ladderSample(epsilon=epsilon, sens=1., M=M, dst=dst, ladders=ladders,
                weights=weights, ranges=ranges, trueCount=trueCount)
        elif (queryType == 'pairsCommonNeighbors') or (queryType == 'pairsTriangleCounts'):
            nodesNum = Gstat.nodesNum

            aliasMethod = AliasMethod(weights)

            noisyCount = defaultdict(float)
            for i in tqdm(range(nodesNum)):
                for j in range(i+1, nodesNum):
                    trueCount = Gstat.geta(i, j)
                    noisyCount['{},{}'.format(i, j)] = self.ladderSample(epsilon=epsilon, sens=sens, M=M, dst=dst, ladders=ladders,
                        weights=weights, ranges=ranges, trueCount=trueCount, aliasMethod=aliasMethod)
        return noisyCount
    
    
    # def fit_triangle(self, epsilon, trueCount=None, ladders=None):
    #     '''计算三角形个数
    #     '''
    #     G, Gstat, queryType = self.G, self.Gstat, self.queryType
    #     if trueCount == None: trueCount = self.count(G, Gstat, queryType)

    #     # ladders: ladder function evaluated on G
    #     if ladders == None: ladders = self.ladderFunction(G, Gstat, queryType)
    #     # M: length of the ladder function
    #     M = len(ladders)

    #     ranges = [0.]
    #     weights = [1.0] # the center's weight

    #     # rungs 1 to M
    #     dst = 0.0
    #     for t in range(M):
    #         weights.append(2*ladders[t]*np.exp(epsilon/2.0*(-t-1)))
    #         dst = dst + ladders[t] # M+1个值，相邻两个值组成范围
    #         ranges.append(dst)
            
    #     # rung M+1
    #     weights.append(2*ladders[-1]*np.exp(epsilon/2.0*(-M-1)) / (1-np.exp(-epsilon/2.0)))
    #     # print('ranges, weights :', ranges,  weights)

    #     ####the only part that involves randomness, may store the earlier result for evaluation over multiple runs
    #     noisyCount = self.ladderSample(epsilon=epsilon, sens=1, M=M, dst=dst, ladders=ladders,
    #         weights=weights, ranges=ranges, trueCount=trueCount)
    #     return noisyCount

    def ladderSample(self, epsilon, sens, M, dst, ladders,
            weights, ranges, trueCount,
            aliasMethod=None):
        '''根据weights和ranges抽样
            使用alias method加速采样
        '''
        noisyCount = 0
        if aliasMethod == None:
            t = int(self._sampleProbList(weights))
        else:
            t = aliasMethod.sample()
        # print('t, len(ranges) = {}, {}, {}'.format(t, len(ranges), len(weights)))
        if t==0:
            noisyCount = trueCount
        elif t <= M:
            flag = -1.0
            if (np.random.uniform() > 0.5):
                flag = 1.0
            low = ranges[t-1]
            delta = np.ceil(np.random.uniform() * (ranges[t] - ranges[t-1]))
            noisyCount = flag * (low + delta) + trueCount
        else:
            p = 1. - np.exp(-epsilon/2./sens)
            ext = np.random.geometric(p)
            low = dst + ext * ladders[-1]
            high = low + ladders[-1]
            flag = -1.
            if (np.random.uniform()>0.5):
                flag = 1.
            noisyCount = flag * np.random.randint(low+1, high+1) + trueCount # low取不到
        return noisyCount
        

    def count(self, G, Gstat: GraphStat, queryType: str):
        '''计数
        '''
        startTime = time.time()
        count = 0
        if queryType == 'triangle':
            for u, v in G.edges():
                count += Gstat.geta(min(u, v), max(u, v))
            count //= 3
        elif queryType == 'pairsCommonNeighbors':
            count = Gstat.a
        elif queryType == 'pairsTriangleCounts':
            # ok, assert False, "pairsTriangleCounts: does not count 'count()'"
            count = defaultdict(int)
            for k, v in Gstat.a.items():
                count[k] = 0
                if G.has_edge(int(k.split(',')[0]), int(k.split(',')[1])): # 如果当前顶点没有边，不可
                    count[k] = v
        print('---------count: {} seconds--------------'.format(time.time() - startTime))
        return count

    def ladderFunction(self, G, Gstat: GraphStat, queryType: str):
        lsd = []
        if queryType == 'triangle':
            lsd = self._lsd_triangle(G, Gstat, queryType)
        elif queryType == 'pairsCommonNeighbors':
            lsd = self._lsd_pairsCommonNeighbors(G, Gstat, queryType)
        elif queryType == 'pairsTriangleCounts':
            lsd = self._lsd_pairsTriangleCounts(G, Gstat, queryType)
        return lsd

    def _lsd_triangle(self, G, Gstat: GraphStat, queryType: str):
        '''ok
        '''
        startTime = time.time()
        nodesNum = Gstat.nodesNum
        bucket = [-1] * nodesNum # bucket: the common neighbor sizes
        for i in tqdm(range(nodesNum)):
            for j in range(i+1, nodesNum):
                aij = Gstat.geta(i, j)
                bij = G.degree[i] + G.degree[j] - 2 * aij - 2 * int(G.has_edge(i, j))
                bucket[aij] = max(bucket[aij], bij)
                # print('bucket :',  bucket[aij], i,  j, aij, bij)

        uppers = []
        for i in reversed(range(nodesNum)):
            if bucket[i] < 0:
                continue
            if (len(uppers) == 0) or (i * 2 + bucket[i] > uppers[-1][0] * 2 + uppers[-1][1]): # 为什么要乘2
                uppers.append([i, bucket[i]])
        print('len(uppers) =', len(uppers))

        gs = self._computeGS(G, Gstat, queryType)
        print('---------_lsd_triangle: {} seconds--------------'.format(time.time() - startTime))

        LSD = []
        t = 0
        while 1:
            lsd = 0
            for p in uppers:
                lsd = max(lsd, p[0]+(t+min(t,p[1]))//2)
            t += 1
            if lsd < gs:
                LSD.append(lsd)
            else:
                LSD.append(gs)
                return LSD

    def _lsd_pairsCommonNeighbors(self, G, Gstat: GraphStat, queryType: str):
        # ok assert False, 'Wrong _lsd_pairsCommonNeighbors'
        nodesNum = Gstat.nodesNum
        # 取前2大的degree值
        bucket = [-1] * 2
        for i in range(nodesNum):
            if G.degree[i] > bucket[0]:
                bucket[1], bucket[0] = bucket[0], G.degree[i]
            elif G.degree[i] > bucket[1]:
                bucket[1] = G.degree[i]
        
        gs = self._computeGS(G, Gstat, queryType)

        LSD = []
        t = 0
        while 1:
            lsd = bucket[0] + bucket[1] + t
            t += 1
            if lsd < gs:
                LSD.append(lsd)
            else:
                LSD.append(gs)
                return LSD

    def _lsd_pairsTriangleCounts(self, G, Gstat: GraphStat, queryType: str):
        LSD = self._lsd_triangle(G, Gstat, 'triangle') # 按照triangle的算GS
        return list(np.array(LSD) * 3)

    def _sampleProbList(self, probList):
        '''从列表中抽样
        '''
        #to-do: 可以用embedding中的方法加速
        normalizedProbList = probList / sum(probList)
        r = np.random.uniform(0, 1)
        s = 0
        for i in range(len(probList)):
            s += normalizedProbList[i]
            if s >= r:
                return i
        return len(probList)-1

    def _computeGS(self, G, Gstat: GraphStat, queryType: str):
        nodeNums = Gstat.nodesNum
        if queryType == 'triangle':
            return nodeNums - 2
        elif queryType == 'pairsCommonNeighbors':
            return 2 * (nodeNums - 2)
        elif queryType == 'pairsTriangleCounts':
            return 3 * (nodeNums - 2)