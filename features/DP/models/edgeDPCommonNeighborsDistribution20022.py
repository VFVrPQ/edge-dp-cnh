'''
模仿EdgeDPCommonNeighborDistributionSequenceThreeStagesLL,把class明确划分下。主要代码还是在edgeDPDegreeDistribution.py下。
'''
from ast import Num
from configparser import NoSectionError
from copy import deepcopy
import numpy as np
import random
import time
import math
import networkx as nx
import scipy.special as sc
from scipy.special import zeta
from scipy.optimize import leastsq
from sklearn.isotonic import IsotonicRegression

from diffprivlib.mechanisms.geometric import GeometricTruncated

import sys
sys.path.append('../../../')
from features.random.randomSelect import sampleProbList
from features.DP.postProcessing import PostProcessing


#############
def unique(seq): 
    '''原先的去重不行，换了
    '''
    # not order preserving 
    # set = {} 
    # map(set.__setitem__, seq, [])
    # return set.keys()
    return list(set(seq))

def hist2seq(hist, xmin, xmax):
    seq = []
    for i in range(xmin, xmax+1):
        if hist[i] > 0:
            seq.extend([i]*int(hist[i]))
    return seq

class EdgeDPCommonNeighborDistributionAnonHist_paper_L1:
    '''edge DP使用laplace计算Common Neighbors' distribution
        分一部分隐私预算截断max number of common neighbors
    '''
    def __init__(self, PP_TOTAL_COUNT='EMPTY'):
        self.PP_TOTAL_COUNT = PP_TOTAL_COUNT # ['EMPTY', 'CLS_total_count']
        pass
    
    def fit(self, n, seq, hist, sens=2, epsilon=1., T=0):
        '''
            n: number of nodes
            hist: common neighbors' distribiton
            sens: senstivity 
            epsilon
        '''
        epsilon1 = epsilon / 10.

        # CN_sum 暂时不变
        CN_sum = 0
        for i in range(len(hist)):
            CN_sum += hist[i] * i
        # 每个位置最大n*(n-1)/2，共有0~n-2个位置。
        g = GeometricTruncated(epsilon=epsilon1, sensitivity=2*(n-2), lower=1, upper=int(n*(n-1)//2*(n-1)))
        noisy_CN_sum = g.randomise(CN_sum)
        noisy_CN_sum1 = 2 * max(1, noisy_CN_sum)

        m = int(math.ceil(math.sqrt(noisy_CN_sum1)))
        m = min(m, n*(n-1)//2) # 上限是所有的点对个数
        
        # add noise
        epsilon2 = epsilon - epsilon1
        ## seq
        noisy_s = seq[:m] + np.random.laplace(loc=0, scale=2*(n-2) / epsilon2, size=m)
        ## post-processing
        ir = IsotonicRegression(y_min=0, y_max=n-2, increasing=False) 
        pp_seq = ir.fit_transform(X=range(len(noisy_s)), y=noisy_s)
        round_seq = np.round(pp_seq) # 变整数

        ## cumhist
        len_hist = min(len(hist), m+1) # 最大的范围；m是最大可能值，m+1是范围
        if len_hist <= T: # 长度不超过T, 全部是seq
            roundCumhist = []
        else:
            temp_hist = np.zeros(shape=(len_hist,), dtype=np.int64)
            for i in range(T, len_hist-1): # 从T开始
                temp_hist[i] = hist[i]
            temp_hist[len_hist-1] = np.sum(hist[len_hist-1:])
            for i in range(m):
                if seq[i] >= T: # >=T的才需要删掉
                    temp_hist[ min(seq[i], len_hist-1) ] -= 1
            cumhist = PostProcessing.pdf2Rcdf(temp_hist) # 翻转cumhist

            noisy_cumhist = cumhist[T:len(cumhist)] + np.random.laplace(loc=0, scale=2*(n-2) / epsilon2, size=len(cumhist)-T)
            zoom_cumhist = noisy_cumhist
            # print('zoom: noisy_CN_sum1 = {}, noisy_cumhist = {}, zoom_cumhist = {}'.format(noisy_CN_sum1, self.sum_ch(cumhist=noisy_cumhist, T=T), self.sum_ch(cumhist=zoom_cumhist, T=T)))
            ir = IsotonicRegression(y_min=0, y_max=n*(n-1)//2-m, increasing=False)
            pp_cumhist = ir.fit_transform(X=range(len(zoom_cumhist)), y=zoom_cumhist)
            roundCumhist = np.round(pp_cumhist)
            if T == 0:
                roundCumhist[0] = n*(n-1)//2-m # 差值全部加到0上

        newhist = [0 for i in range(T)]
        newhist.extend(PostProcessing.rcdf2Pdf(roundCumhist))
        newhist.extend([0 for i in range(max(T, len_hist), n-2+1)])
        for i in range(len(round_seq)):
            if int(round_seq[i]) >= T:
                newhist[ int(round_seq[i]) ] += 1

        newhist_sum = 0
        for i in range(len(newhist)):
            newhist_sum += newhist[i] * i # noisy_total_count
        print('AnonHist: m = {}, sum(noisy_seq) = {}, sum(seq[:m]) = {}, noisy_total_count = {}, total_count = {}, noisy_CN_sum1 = {}'.format(
            m, np.linalg.norm(round_seq, ord=1), np.sum(seq[:m]), newhist_sum, CN_sum, noisy_CN_sum1))

        if self.PP_TOTAL_COUNT == 'CLS_total_count':
            pphist = PostProcessing.CLS_total_count(newhist, noisy_CN_sum1)
        elif self.PP_TOTAL_COUNT == 'scaling_total_count':
            pphist = PostProcessing.scaling_total_count(newhist, noisy_CN_sum1)
        else:
            pphist = newhist

        self.r_chosen = m
        self.tau_chosen = len_hist-1 # maximum count of common neighbors in the long tail
        return pphist

    def zoom(self, noisy_cumhist, T, sum):
        '''noisy_cumhist的总和noisy放缩到sum, noisy_cumhist下标为0表示值为T
        '''
        noisy_hist = PostProcessing.rcdf2Pdf(noisy_cumhist)
        noisySum = 0
        for i in range(len(noisy_hist)):
            noisySum += noisy_hist[i] * (i + T)
        ratio = sum / noisySum
        if ratio >= 1:
            return noisy_cumhist
        else: # 只有太大了才需要限制
            for i in range(len(noisy_hist)):
                noisy_hist[i] = noisy_hist[i] * ratio
            return PostProcessing.pdf2Rcdf(noisy_hist)
    
    def sum_ch(self, cumhist, T):
        hist = PostProcessing.rcdf2Pdf(cumhist)
        noisySum = 0
        for i in range(len(hist)):
            noisySum += hist[i] * (i + T)
        return noisySum


class EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022:
    '''edge DP使用laplace和laplace计算Common Neighbors' distribution，三阶段
    '''
    def __init__(self, getChooseRTauParam = 'r_then_tau', getPowerlawIntervalParam='N', getSeqParam='Laplace', SAMPLE=500, fixed_r=100, getStepParam=10):
        '''SAMPLE是计算pvalue时的采样个数
        '''
        self._getChooseRTauParam = getChooseRTauParam # ['r_and_tau', 'r_then_tau', 'r_then_tau_fixed_r']
        self._getPowerlawIntervalParam = getPowerlawIntervalParam #['Y', 'N', 'Y_partial', 'Y_partial_confirm', 'Y_partial_confirm_fixed_step']
        self._getSeqParam = getSeqParam # ['Laplace', 'EEM', 'EEM_pure', 'rEEM']
        self._getSampleParam = SAMPLE
        self._fixed_r = fixed_r
        self._getStepParam = getStepParam # fixedStep
        # print('get_: chooseRTauParam = {}, powerlawIntervalParam = {}, seqParam = {}, SAMPLE = {}, fixed_r = {}, getStepParam = {}'.format( 
        #     self._getChooseRTauParam, self._getPowerlawIntervalParam, self._getSeqParam, self._getSampleParam, self._fixed_r, self._getStepParam))
    
    def fit(self, n, seq, hist=None, maxPartition=500, sens=2, epsilon=1., budgetAllocation=None, T=0):
        '''
            n: number of nodes
            seq: common neighbor sequence [non-increasing]，共有n(n-1)/2个值
            maxPartition: 最大分割点的位置，而后使用指数机制选择位置
            sens: senstivity 
            epsilon
            T: the minimum count of common neighbors 
        '''
        if budgetAllocation == None: # 主要是epsilon1的分配
            epsilon1 = epsilon / 10.  # 求分割点的值
            epsilon2 = epsilon / 10. # seq
        else:
            epsilon1 = epsilon * budgetAllocation
            epsilon2 = (epsilon - epsilon1) / 10.
        epsilon3 = epsilon - epsilon1 - epsilon2  # cumhist


        maxPartition = min(int(np.ceil(((2* epsilon2 / epsilon) ** 0.5) * (n-1) )), len(seq)-1)  # 修改max partition
        # Exponential Mechanism choose (r, tau)
        r_chosen, tau_chosen, MCN_chosen = self.choose_r_and_tau(maxPartition, seq, n, epsilon1, epsilon2, epsilon3, hist, minCount=T)
        # 对epsilon2和epsilon3进行调整
        if (self._getChooseRTauParam == 'r_then_tau_epsilon2') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_trueParam') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_fix'):
            if tau_chosen < T:
                epsilon2 = epsilon - epsilon1
                epsilon3 = 0
            else:
                epsilon2 = r_chosen / (2. * (n-2)) * (epsilon - epsilon1)
                epsilon3 = epsilon - epsilon1 - epsilon2  # cumhist
            print('({}) epsilon2={}, epsilon3={}'.format(self._getChooseRTauParam, epsilon2, epsilon3))
        elif (self._getChooseRTauParam == 'r_then_tau_epsilon2_new') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new_OBA') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_threeStages') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new_fix'):
            if tau_chosen < T:
                epsilon2 = epsilon - epsilon1
                epsilon3 = 0
            else:
                epsilon2 = (epsilon - epsilon1) * r_chosen / (r_chosen + math.sqrt((tau_chosen-T+1)*(2*(n-2)-r_chosen)))
                epsilon3 = epsilon - epsilon1 - epsilon2  # cumhist
            print('(r_then_tau_epsilon2_new) epsilon2={}, epsilon3={}'.format(epsilon2, epsilon3))

        noisySeq1 = self.dpForSeq(seq=seq, n=n, epsilon=epsilon2, r_chosen=r_chosen, 
                tau_chosen=tau_chosen, MCN=MCN_chosen) # MCN_chosen调整

        # 2-2. laplace
        if epsilon3 > 0:
            if self._getChooseRTauParam == 'r_then_tau_new' or (self._getChooseRTauParam == 'r_then_tau_epsilon2') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_trueParam') \
                or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new_trueParam') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new_OBA') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_threeStages') \
                    or (self._getChooseRTauParam == 'r_then_tau_new_fix') or (self._getChooseRTauParam == 'r_then_tau_epsilon2_new_fix'):
                epsilon3 = epsilon3 * 2*(n-2) / (2*(n-2) - r_chosen) # 隐私预算放大
            noisyHist2 = self.laplaceForCumulativeHistogram(seq=seq, n=n, partition=r_chosen, noisyPartitionValue=tau_chosen, epsilon=epsilon3, trueHist=hist, minCount=T)
        else:
            noisyHist2 = []
            
        # histogram
        dd = [0 for _ in range(n-2+1)] # 最大n-2
        for i in range(r_chosen):
            if int(noisySeq1[i]) >= T:
                dd[ int(noisySeq1[i])] += 1 
        for i in range(len(noisyHist2)):
            dd[i] += noisyHist2[i]
        pp_dd = dd


        CN_sum = 0
        for i in range(len(hist)):
            CN_sum += hist[i] * i
        newhist_sum = 0
        for i in range(len(dd)):
            newhist_sum += dd[i] * i
        print('sum(noisy_seq) = {}, sum(seq[:m]) = {}, noisy_total_count = {}, total_count = {}'.format(
            np.linalg.norm(noisySeq1, ord=1), np.sum(seq[:r_chosen]), newhist_sum, CN_sum))
        return pp_dd


    def choose_r_and_tau(self, maxPartition, seq, n, epsilon1, epsilon2, epsilon3, epsilon_threshold=2, hist=None, minCount=0):
        '''Exponential Mechanism choose (r, tau)
            epsilon2: seq
            epsilon3: cumhist
        '''
        MCN_chosen = n - 2 # n-2 when no budget
        if self._getChooseRTauParam == 'r_and_tau':
            epsilon11 = epsilon1 / 2
            epsilon12 = epsilon1 - epsilon11
            if epsilon11 > epsilon_threshold:
                g = GeometricTruncated(epsilon=epsilon11, sensitivity=1, lower=1, upper=n-2) # 最小是1
                noisyMCN = g.randomise(seq[0])
            else:
                noisyMCN = n - 2
                epsilon12 += epsilon11

            cumhist = self.calcCumulativeHistogram(seq=seq, partition=0, n=n, MCN=noisyMCN)
            cumcumhist = np.cumsum(cumhist, dtype=np.int64)

            us = []
            uParams = []
            for r in range(maxPartition+1):
                for tau in range(noisyMCN+1): # approx rank r的count
                    sr = min(seq[r], noisyMCN) # true rank r的count
                    u_cumcumhist = 0
                    if sr >= tau:
                        u_cumcumhist = cumcumhist[sr] - cumcumhist[tau]
                    u = - ((r ** 2) / epsilon2 + (tau + 1) * 2 * (n-2) / epsilon3 + u_cumcumhist)
                    # sens = 2 * (n-2)
                    sens = (n*n + 3*n - 8) / 2
                    us.append(epsilon12 * u / (2 * sens))
                    uParams.append((tau, r))

                # 减去seq[r]
                for i in range(seq[r], noisyMCN+1):
                    cumhist[i] -= 1 # cumhist每个都加了1
                    cumcumhist[i] -= (i - seq[r] + 1)
                
            us = np.exp(us)
            uIndex = sampleProbList(us)
            tau_chosen, r_chosen = uParams[uIndex]
            print('MCN, noisyMCN, r_chosen, seq[r], tau_chosen, us[uIndex] =', seq[0], noisyMCN, r_chosen, seq[r_chosen], tau_chosen, us[uIndex])
        elif self._getChooseRTauParam == 'r_then_tau_epsilon2':
            '''[SBA]
            '''
            epsilon11 = epsilon1 / 3 # EM取r
            epsilon12 = epsilon1 / 3 # tau
            epsilon13 = epsilon1 - epsilon11 - epsilon12 # MCN

            maxPartition = min(int(n), len(seq)-1)
            us = []
            for r in range(maxPartition+1):
                tau = seq[r] # change 2022.03.23
                if tau < minCount:
                    u = - r * r / (epsilon2 + epsilon3)
                else:
                    u = - (r + tau - minCount + 1) * (2 * (n - 2) / (epsilon2 + epsilon3))
                sens = 2 * (n - 2) / (epsilon2 + epsilon3)
                us.append(epsilon11 * u / sens) # monotonic
            
            us = np.exp(us)
            r_chosen = sampleProbList(us)

            g = GeometricTruncated(epsilon=epsilon12, sensitivity=1, lower=0, upper=n-2) # 最小是1
            tau_chosen = g.randomise(seq[r_chosen])
            g = GeometricTruncated(epsilon=epsilon13, sensitivity=1, lower=0, upper=n-2) # 最小是1
            MCN_chosen = g.randomise(seq[0])
            if MCN_chosen < tau_chosen:
                MCN_chosen, tau_chosen = tau_chosen, MCN_chosen
            MCN_chosen = max(MCN_chosen, minCount)
            print('(r_then_tau_epsilon2) MCN, MCN_chosen, r_chosen, seq[r_chosen], tau_chosen, us[r_chosen] =', seq[0], MCN_chosen, r_chosen, seq[r_chosen], tau_chosen, us[r_chosen])
        elif self._getChooseRTauParam == 'r_then_tau_epsilon2_fix':
            '''[SBA_fix]
            '''
            epsilon11 = epsilon1 / 3 # EM取r
            epsilon12 = epsilon1 / 3 # tau的
            epsilon13 = epsilon1 - epsilon11 - epsilon12 # MCN的

            r_chosen = self._fixed_r

            g = GeometricTruncated(epsilon=epsilon12, sensitivity=1, lower=0, upper=n-2) # 最小是1
            tau_chosen = g.randomise(seq[r_chosen])
            g = GeometricTruncated(epsilon=epsilon13, sensitivity=1, lower=0, upper=n-2) # 最小是1
            MCN_chosen = g.randomise(seq[0])
            if MCN_chosen < tau_chosen:
                MCN_chosen, tau_chosen = tau_chosen, MCN_chosen
            MCN_chosen = max(MCN_chosen, minCount)
            print('(r_then_tau_epsilon2_fix) MCN, MCN_chosen, r_chosen, seq[r_chosen], tau_chosen =', seq[0], MCN_chosen, r_chosen, seq[r_chosen], tau_chosen)
        elif self._getChooseRTauParam == 'r_then_tau_epsilon2_new':
            '''[OBA]
            '''
            epsilon11 = epsilon1 / 3 # EM取r
            epsilon12 = epsilon1 / 3 # tau的
            epsilon13 = epsilon1 - epsilon11 - epsilon12 # MCN

            us = []
            # for r in range(maxPartition+1):
            for r in range(int((math.sqrt(1+8*(n-2))-1)//2)+1):
                tau = seq[r] # change 2022.03.23
                if tau < minCount:
                    u = - r * r / (epsilon2 + epsilon3)
                else:
                    u = - (r + math.sqrt((tau - minCount + 1) * (2 * (n - 2) - r))) ** 2
                sens = 4 * (n - 2)
                us.append(epsilon11 * u / sens) # monotonic
            
            us = np.exp(us)
            r_chosen = sampleProbList(us)

            g = GeometricTruncated(epsilon=epsilon12, sensitivity=1, lower=0, upper=n-2) # 最小是1
            tau_chosen = g.randomise(seq[r_chosen])
            g = GeometricTruncated(epsilon=epsilon13, sensitivity=1, lower=0, upper=n-2) # 最小是1
            MCN_chosen = g.randomise(seq[0])
            if MCN_chosen < tau_chosen:
                MCN_chosen, tau_chosen = tau_chosen, MCN_chosen
            MCN_chosen = max(MCN_chosen, minCount)
            print('(r_then_tau_epsilon2_new) MCN, MCN_chosen, r_chosen, seq[r_chosen], tau_chosen, us[r_chosen] =', seq[0], MCN_chosen, r_chosen, seq[r_chosen], tau_chosen, us[r_chosen])
        elif self._getChooseRTauParam == 'r_then_tau_epsilon2_new_fix':
            '''[OBA-fix]
            '''
            epsilon11 = epsilon1 / 3 # EM取r
            epsilon12 = epsilon1 / 3 # tau的
            epsilon13 = epsilon1 - epsilon11 - epsilon12 # MCN

            r_chosen = self._fixed_r

            g = GeometricTruncated(epsilon=epsilon12, sensitivity=1, lower=0, upper=n-2) # 最小是1
            tau_chosen = g.randomise(seq[r_chosen])
            g = GeometricTruncated(epsilon=epsilon13, sensitivity=1, lower=0, upper=n-2) # 最小是1
            MCN_chosen = g.randomise(seq[0])
            if MCN_chosen < tau_chosen:
                MCN_chosen, tau_chosen = tau_chosen, MCN_chosen
            MCN_chosen = max(MCN_chosen, minCount)
            print('(r_then_tau_epsilon2_new_fix) MCN, MCN_chosen, r_chosen, seq[r_chosen], tau_chosen =', seq[0], MCN_chosen, r_chosen, seq[r_chosen], tau_chosen)
        self.r_chosen = r_chosen
        self.tau_chosen = tau_chosen
        self.MCN_chosen = MCN_chosen
        return r_chosen, tau_chosen, MCN_chosen

    def getRtau(self):
        return self.r_chosen, self.tau_chosen

    def dpForSeq(self, seq, n, epsilon, r_chosen, tau_chosen, MCN):
        '''
            seq长度是r_chosen,取值范围[tau_chosen, MCN]
            seq是递降的
        '''
        noisySeq1 = []
        if r_chosen > 0: # 有值才考虑运算
            if self._getSeqParam == 'Laplace':
                noisySeq = seq[:r_chosen] + np.random.laplace(loc=0, scale=min(r_chosen, 2*(n-2)) / epsilon, size=r_chosen)
                ir = IsotonicRegression(y_min=tau_chosen, y_max=MCN, increasing=False) 
                ppSeq1 = ir.fit_transform(X=range(len(noisySeq)), y=noisySeq)
                roundPPSeq1 = np.round(ppSeq1) # 变整数
                noisySeq1 = roundPPSeq1
            elif self._getSeqParam == 'EEM':
                '''长度为r_chosen,每个值的取值范围是[tau_chosen, noisy_MCN]
                    计算DP是从max端开始, 抽样我是从min端抽样。结果是max端稠密, min端稀疏。
                '''
                # 1. 求取值范围
                epsilon1 = epsilon / 10.
                epsilon2 = epsilon - epsilon1

                gt = GeometricTruncated(epsilon=epsilon1, sensitivity=1, lower=1, upper=n-2)
                noisyMCN = gt.randomise(seq[0])
                noisyTau = tau_chosen
                if noisyMCN < noisyTau:
                    noisyMCN, noisyTau = noisyTau, noisyMCN # 交换彼此

                # 2. EEM
                d = r_chosen + 1 # r+1个值,考虑多一个值作为最大值,以方便运算
                maxj = noisyMCN - noisyTau # 每个值得范围变换到[0, noisyMCN - noisyTau]
                curSeq = np.array(list(reversed(seq[:r_chosen+1]))) # 
                curSeq = curSeq - noisyTau # test ok

                sens = min(r_chosen, 2*(n-2))
                self.dynamicProgramming(n=d, maxj=maxj, epsilon=epsilon2, sensitivity=sens, seq=curSeq)
                noisySeq1 = self.getNoisyOuput(n=d, maxj=maxj) + noisyTau 


                # 是递增的,重新转换为递降的
                noisySeq1 = list(reversed(noisySeq1))[:-1] # 多的值要还回来
                print('EEM : len(noisySeq1) = {}, r_chosen = {}, noisyMCN = {}, noisyTau = {}'.format(len(noisySeq1), r_chosen, noisyMCN, noisyTau))
                print('EEM : noisySeq1[:20] = {}, seq[:20] = {}'.format(noisySeq1[:20], seq[:20]))
                print('EEM : noisySeq1[-20:] = {}, seq[:] = {}'.format(noisySeq1[-20:], seq[max(r_chosen-20+1, 0):r_chosen+1]))
            elif self._getSeqParam == 'rEEM': 
                '''长度为r_chosen,每个值的取值范围是[tau_chosen, noisy_MCN]
                    反向抽样， 计算DP是从min端开始, 抽样从max端抽样。期望结果是min端稠密, max端稀疏。
                '''
                # 1. 求取值范围
                # epsilon1 = epsilon / 3.
                # epsilon2 = epsilon - epsilon1
                epsilon1 = epsilon / 10.
                epsilon2 = epsilon - epsilon1

                gt = GeometricTruncated(epsilon=epsilon1, sensitivity=1, lower=1, upper=n-2)
                noisyMCN = gt.randomise(seq[0])
                noisyTau = tau_chosen
                if noisyMCN < noisyTau:
                    noisyMCN, noisyTau = noisyTau, noisyMCN # 交换彼此

                # 2. rEEM
                d = r_chosen # r个值,不考虑多一个值
                maxj = noisyMCN - noisyTau # 每个值得范围变换到[0, noisyMCN - noisyTau]
                curSeq = np.array(list(reversed(seq[:r_chosen]))) # 
                curSeq = curSeq - noisyTau # test ok

                sens = min(r_chosen, 2*(n-2))
                self.dynamicProgramming_reversed(n=d, maxj=maxj, epsilon=epsilon2, sensitivity=sens, seq=curSeq)
                noisySeq1 = self.getNoisyOuput_reversed(n=d, maxj=maxj) + noisyTau 

                # print(self.f)
                # 是递增的,重新转换为递降的
                noisySeq1 = list(reversed(noisySeq1))
                print('rEEM : len(noisySeq1) = {}, r_chosen = {}, noisyMCN = {}, noisyTau = {}'.format(len(noisySeq1), r_chosen, noisyMCN, noisyTau))
                print('rEEM : noisySeq1[:20] = {}, seq[:20] = {}'.format(noisySeq1[:20], seq[:20]))
                print('rEEM : noisySeq1[-20:] = {}, seq[:] = {}'.format(noisySeq1[-20:], seq[max(r_chosen-20+1, 0):r_chosen+1]))
            elif self._getSeqParam == 'EEM_pure':
                '''长度为r_chosen,每个值的取值范围是[tau_chosen, noisy_MCN]
                    noisyMCN直接取n-2。
                    想看看jointEM是否会取到n-2, 还是会有一定上限
                '''
                # 1. 求取值范围
                noisyMCN = n - 2
                noisyTau = tau_chosen

                # 2. EEM
                d = r_chosen + 1 # r+1个值,考虑多一个值作为最大值,以方便运算
                maxj = noisyMCN - noisyTau # 每个值得范围变换到[0, noisyMCN - noisyTau]
                curSeq = np.array(list(reversed(seq[:r_chosen+1]))) # 
                curSeq = curSeq - noisyTau # test ok

                sens = min(r_chosen, 2*(n-2))
                self.dynamicProgramming(n=d, maxj=maxj, epsilon=epsilon, sensitivity=sens, seq=curSeq)
                noisySeq1 = self.getNoisyOuput(n=d, maxj=maxj) + noisyTau 
                

                # 是递增的,重新转换为递降的
                noisySeq1 = list(reversed(noisySeq1))[:-1] # 多的值要还回来
                print('EEM_pure : len(noisySeq1) = {}, r_chosen = {}, noisyMCN = {}, noisyTau = {}'.format(len(noisySeq1), r_chosen, noisyMCN, noisyTau))
                print('EEM_pure : noisySeq1[:20] = {}, seq[:20] = {}'.format(noisySeq1[:20], seq[:20]))
                print('EEM_pure : noisySeq1[-20:] = {}, seq[-20:] = {}'.format(noisySeq1[-20:], seq[max(r_chosen-20+1, 0):r_chosen+1]))
        return noisySeq1

    def dynamicProgramming(self, n, maxj, epsilon, sensitivity, seq):
        '''求f,sf

            n表示长度, maxj表示seq最大值
        '''
        self.f = np.zeros(shape=(n, maxj+1), dtype=float) # f[k][i]:第一维是第k位，第二维表示输出为i
        sf = np.zeros(shape=(n, maxj+1), dtype=float) # 辅助数组
        
        # 边界
        self.f[n-1][maxj] = 1
        for i in range(maxj, -1, -1):
            sf[n-1][i] = 1
        temp = np.array([i for i in range(0, maxj+1)])

        for k in range(n-2, -1, -1):    
            self.f[k] = np.exp(epsilon * (-abs(temp - seq[k])) / 2. / sensitivity) * sf[k+1]
            self.f[k] = self.f[k] / sum(self.f[k]) # 每次都归一化，不会影响最终结果
            # print('self.f[k].copy()', self.f[k].copy())
            sf[k] = self.f[k].copy()
            sf[k] = list(reversed(np.cumsum(list(reversed(sf[k])))))

    def getNoisyOuput(self, n, maxj):
        '''依次按如下概率抽样每个值, 得到输出序列
        '''
        lastValue = 0
        t = []
        for k in range(n):
            nxt = lastValue + sampleProbList([self.f[k][i] for i in range(lastValue, maxj+1)]) # f[k][i], i<=n
            t.append(nxt)
            lastValue = nxt
        
        # c = PostProcessing.cdf2Pdf(t)
        return np.asarray(t)

    def dynamicProgramming_reversed(self, n, maxj, epsilon, sensitivity, seq):
        '''求f,sf
            seq是从小到大
            n表示长度, maxj表示seq最大值
            反向抽样， 计算DP是从min端开始, 抽样从max端抽样。期望结果是min端稠密, max端稀疏。
        '''
        self.f = np.zeros(shape=(n, maxj+1), dtype=float) # f[k][i]:第一维是第k位，第二维表示输出为i
        sf = np.zeros(shape=(n, maxj+1), dtype=float) # 辅助数组

        # 边界
        temp = np.array([i for i in range(0, maxj+1)])
        self.f[0] = np.exp(epsilon * (-abs(temp - seq[0])) / 2. / sensitivity)
        self.f[0] = self.f[0] / np.sum(self.f[0])
        sf[0] = np.cumsum(self.f[0])

        for k in range(1, n):
            '''当前取到v的时候, 前面的可以取0~v
            '''
            self.f[k] = np.exp(epsilon * (-abs(temp - seq[k])) / 2. / sensitivity) * sf[k-1]
            self.f[k] = self.f[k] / sum(self.f[k]) # 每次都归一化，不会影响最终结果
            sf[k] = np.cumsum(self.f[k])

    def getNoisyOuput_reversed(self, n, maxj):
        '''依次按如下概率抽样每个值, 得到输出序列
            从大到小抽样, 返回的时候还是从小到大
        '''
        lastValue = maxj
        t = []
        for k in range(n-1, -1, -1):
            nxt = sampleProbList([self.f[k][i] for i in range(0, lastValue+1)]) # f[k][i], i<=n
            t.append(nxt)
            lastValue = nxt
        
        return np.asarray(list(reversed(t)))

    def calcCumulativeHistogram(self, seq, n, partition, MCN):
        '''
            partition: 从partition开始统计
            MCN: maximum number of common neighbors；保证seq中所有值都是小于MCN的
        '''
        hist = [0 for i in range(MCN+1)]
        for i in range(partition, len(seq)):
            hist[ min(seq[i], MCN) ] += 1 # 保证seq中所有值都是小于MCN的
        
        cumhist = np.cumsum(hist)
        return cumhist

    def laplaceForCumulativeHistogram(self, seq, n, partition, noisyPartitionValue, epsilon, trueHist=None, minCount=0):
        '''laplace噪声加到cumulative histogram上。
            cumulative histogram取值范围是[0, noisyPartitionValue]，个数是n(n-1)//2-partition
            返回hist
        '''
        hist = [0 for i in range(noisyPartitionValue+1)]
        if str(type(trueHist)) == "<class 'numpy.ndarray'>":
            # 先全部转移过来，再一个一个删除\
            for i in range(minCount, noisyPartitionValue): # 只能转移minCount， 
                hist[i] = trueHist[i]
            hist[noisyPartitionValue] = np.sum(trueHist[noisyPartitionValue:]) 
            for i in range(partition): #一个一个扔
                if seq[i] >= minCount:
                    hist[ min(seq[i], noisyPartitionValue) ] -= 1
        else:
           for i in range(partition, len(seq)):
                # if seq[i] <= noisyPartitionValue: # 超过的直接扔掉，这部分的误差已经算在utility里面了【这步是有问题的。】
                if seq[i] >= minCount:
                    hist[ min(seq[i], noisyPartitionValue) ] += 1 # 不能扔掉
        cumhist = PostProcessing.pdf2Rcdf(hist) # 翻转hist
        noisyCumhist = cumhist[minCount:] + np.random.laplace(loc=0, scale=2*(n-2) / epsilon, size=noisyPartitionValue+1-minCount)
        ir = IsotonicRegression(y_min=0, y_max=n*(n-1)//2-partition, increasing=False) # post-processing
        ppCumhist = ir.fit_transform(X=range(len(noisyCumhist)), y=noisyCumhist)
        if minCount == 0: # 2022.06.15
            ppCumhist[0] = n*(n-1)//2-partition # # 差值全部加到minCount上, 2022.06.02 修改
        roundCumhist = np.round(ppCumhist)

        newhist = [0 for i in range(minCount)]
        newhist.extend( PostProcessing.rcdf2Pdf(roundCumhist) )
        return newhist


def int2log(hist):
    '''int变成log10(int)，=0时变为0
    '''
    loghist = []
    for x in hist:
        if x <= 0:
            loghist.append(0)
        else:
            loghist.append(np.log10(x))
    return loghist

def sqrError(p, x, y):
    '''x, y是单个值
    '''
    f = np.poly1d(p)
    res = (f(x) - y) ** 2
    return res

