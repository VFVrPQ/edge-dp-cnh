import numpy as np
import random
import time
import networkx as nx
from sklearn.isotonic import IsotonicRegression

from diffprivlib.mechanisms.geometric import GeometricTruncated

import sys
sys.path.append('../../../')
from features.random.randomSelect import sampleProbList
from features.DP.postProcessing import PostProcessing

#=====================

'''common neighbors' histogram
'''
class EdgeDPCommonNeighborDistributionLaplaceCumulateHistogram:
    '''edge DP使用laplace计算Common Neighbors' distribution
        LMCH
    '''
    def __init__(self):
        pass
    
    def fit(self, n, b, sens=2, epsilon=1., T=0):
        '''
            n: number of nodes
            b: common neighbor histogram
            sens: senstivity 
            epsilon
            T: minimum count
        '''
        if T>n-2: # 大家都知道没值
            c = [0 for i in range(len(b))]
        else:
            # 只处理>=T的
            sens = 2 * (n-2)
            s = PostProcessing.pdf2Rcdf(b[T:])
            t = s + np.random.laplace(loc=0, scale=sens/epsilon, size=len(s))
            ir = IsotonicRegression(y_min=0, y_max=n*(n-1)//2, increasing=False) # 不知道最大值具体是多少，只能用n(n-1)/2估计
            t = ir.fit_transform(X=range(len(t)), y=t)
            t = np.round(t) # 变成整数
            if T == 0: # 除了0以外, 最大值不确定了
                t[0] = n*(n-1)//2 # 2022.06.02, 差值全部加到0上; 2022.06.15翻转
            c = [0 for i in range(T)]
            c.extend(PostProcessing.rcdf2Pdf(t))
        return c



class EdgeDPCommonNeighborDistributionLaplaceCumulateHistogramTruncated:
    '''edge DP使用laplace计算Common Neighbors' distribution

        分一部分隐私预算截断max number of common neighbors
    '''
    def __init__(self):
        pass
    
    def fit(self, n, b, sens=2, epsilon=1., epsilon1=None, budgetAllocation=None, T=0):
        '''n: number of nodes
            b: common neighbors' distribiton【】
            sens: senstivity 
            epsilon
            T: the minimum count of common neighbors
        '''
        if budgetAllocation == None:
            epsilon1 = epsilon / 10. * 1 # 用于MCN
        else:
            epsilon1 = epsilon * budgetAllocation
        epsilon2 = epsilon - epsilon1 # 用于cum hist
        
        # noisy MCN
        MCN = 0 # max common neighbors
        for i in range(len(b)):
            if b[i] > 0:
                MCN = max(MCN, i)
        g = GeometricTruncated(epsilon=epsilon1, sensitivity=1, lower=T, upper=n-2)
        noisyMCN = g.randomise(MCN)
        print('epsilon1 = {}, minCount = {}, MCN = {}, noisyMCN = {}'.format(epsilon1, T, MCN, noisyMCN))

        # noisy Cumhist
        cumhist = PostProcessing.pdf2Rcdf(b) # cumhist
        sens = 2 * (n-2)
        t = cumhist[T:noisyMCN+1] + np.random.laplace(loc=0, scale=sens/epsilon2, size=noisyMCN+1-T) # 长度有限
        # ir = IsotonicRegression(y_min=0, y_max=n*(n-1)//2, increasing=True)
        # pp_t = list(reversed( ir.fit_transform(X=range(len(t)), y=list(reversed(t))) ))
        ir = IsotonicRegression(y_min=0, y_max=n*(n-1)//2, increasing=False)
        pp_t = ir.fit_transform(X=range(len(t)), y=t)
        if T == 0:
            pp_t[0] = n*(n-1)//2 # 2022.06.02, 差值全部加到0上
        pp_t = np.round(pp_t) # 变成整数
        
        # return
        noisyHist = [0 for i in range(T)]
        noisyHist.extend(PostProcessing.rcdf2Pdf(pp_t))
        for i in range(noisyMCN+1, len(b)):
            noisyHist.append(0)
        return noisyHist
