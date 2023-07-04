# 有别于010-01-2，该文件test T不在alg中体现，仅在计算EMD时有体现。
import sys
import os
import networkx as nx
import numpy as np

sys.path.append('../../')
from features.common.constants import DATA_NAMES, DATASET_LIST

from features.DP.models.edgeDP import GraphStatCommonNeighbors
from features.DP.models.edgeDPDegreeDistribution import EdgeDPCommonNeighborDistributionLaplaceCumulateHistogram, EdgeDPCommonNeighborDistributionLaplaceCumulateHistogramTruncated
from features.DP.models.edgeDPCommonNeighborsDistribution20022 import EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022
from features.DP.models.edgeDPCommonNeighborsDistribution20022 import EdgeDPCommonNeighborDistributionAnonHist_paper_L1

from features.common.metrics import DistributionMetrics


def hist2seq(hist):
    seq = []
    for i in range(len(hist)):
        if hist[i] > 0:
            seq.extend([i]*np.int64(hist[i]))
    return seq

#graph transformation/clean up for subgraph counting aglo (e.g. ladder function) 
#this remap the node id, such that node id starts from 0 and increments to the total number of nodes 
def translate(datafile, newDatafile):
    nodeMap = dict()
    
    fin = open(datafile, "r")
    fout = open(newDatafile, "w")
    for ij in fin:
        if ij[0] == '#': continue
        i,j = ij.split()
        #i = int(i)
        #j = int(j)
        if i not in nodeMap:
            nodeMap[i] = len(nodeMap)
        if j not in nodeMap:
            nodeMap[j] = len(nodeMap)
        
        i_map = nodeMap[i]
        j_map = nodeMap[j]
        if i_map < j_map:
            fout.write(str(i_map)+" "+str(j_map)+"\n")
        else:
            fout.write(str(j_map)+" "+str(i_map)+"\n")

    fout.close()
    fin.close() 

# caller
def caller(dataKey, queryKey, partition=None):
    '''
        partition是专门为twoStage设置的
    '''
    dataDir ="../2001.ladder/Datasets/"

    dataNames = DATA_NAMES
    dataName = dataNames[dataKey]
    print(dataName)

    datafile = dataDir+dataName #"facebook_combined.txt"
    if not os.path.isfile(datafile+"-map.txt"):
        translate(datafile+".txt", datafile+"-map.txt")
        #convert all nodes id to 0 to nodeNum
    else:
        print("file exists")


    queryTypeList = ["KSTest", 'EMD']
    queryType = queryTypeList[queryKey]
    print('queryType = {}'.format(queryType))

    algoNames = [
        'EdgeDPCommonNeighborDistributionAnonHist_paper_L1',
        'threeStagesLL2_r_then_tau_epsilon2',
        'threeStagesLL2_r_then_tau_epsilon2_new',
        # 'LMCH',
        'LMCHTruncated'
    ]
    TList = [0, 10, 20, 30, 40, 50]
    TList = [i*5 for i in range(41)]
    repeats = 30
    epsilon = 1
    print('repeats = {}'.format(repeats))


    #######code for utility evaluation########

    if not os.path.isfile(datafile+'-histseq.pickle'):

        print(datafile + "-map.txt")

        G = nx.read_edgelist(datafile + "-map.txt", nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))

        gs = GraphStatCommonNeighbors(G)
        nodesNum, edgesNum, MCN = len(G.nodes()), len(G.edges()), gs.maxA #assume nodesNum is given

        hist = gs.hist
        seq = hist2seq(gs.hist)
        seq.sort(reverse=True)

        import pickle
        file = open(datafile+'-histseq.pickle', 'wb')
        pickle.dump({'hist': hist, 'seq': seq, 'nodesNum': nodesNum, 'edgesNum': edgesNum, 'MCN': MCN}, file)
        file.close()
    else:
        import pickle
        file = open(datafile+'-histseq.pickle', 'rb')
        data = pickle.load(file)
        file.close()

        seq, hist, nodesNum, edgesNum, MCN = data['seq'], data['hist'], data['nodesNum'], data['edgesNum'], data['MCN']

    print('n = {}, edges = {}, max common neighbors = {}'.format(nodesNum, edgesNum, MCN))
    print('hist[:20] =', np.asarray(hist)[:20])
    print('seq[:20], seq[-20:] =', seq[:20], seq[-20:])
    # print('seq =', seq)

    errorsList = [] # 每个算法的位置留好
    histEMDErrorsList = []
    L1ErrorsList = []
    KSErrorsList = []
    for i in range(len(algoNames)):
        errorsList.append([])
        histEMDErrorsList.append([])
        L1ErrorsList.append([])
        KSErrorsList.append([])

    for algoKey in range(len(algoNames)):
        algo = algoNames[algoKey]
        print(algo)

        for T in TList:
            errorsList[algoKey].append([])
            histEMDErrorsList[algoKey].append([])
            L1ErrorsList[algoKey].append([])
            KSErrorsList[algoKey].append([])

        for i in range(repeats):
            if algo == 'LMCH':
                lc = EdgeDPCommonNeighborDistributionLaplaceCumulateHistogram()
                noisy_dd = lc.fit(n=nodesNum, b=hist, sens=2*(nodesNum-2), epsilon=epsilon, T=0)
            elif algo == 'LMCHTruncated':
                lc = EdgeDPCommonNeighborDistributionLaplaceCumulateHistogramTruncated()
                noisy_dd = lc.fit(n=nodesNum, b=hist, sens=2*(nodesNum-2), epsilon=epsilon, T=0)
            elif algo == 'threeStagesLL2_r_then_tau_epsilon2': # SBA
                tll = EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022(getChooseRTauParam='r_then_tau_epsilon2')
                maxPartition = min(int(np.ceil((2 ** 0.5) * nodesNum )), len(seq)-1) # 暂时设定为n
                noisy_dd = tll.fit(n=nodesNum, seq=seq, maxPartition=maxPartition, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=0)
            elif algo == 'threeStagesLL2_r_then_tau_epsilon2_new': #OBA
                tll = EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022(getChooseRTauParam='r_then_tau_epsilon2_new')
                maxPartition = min(int(np.ceil((2 ** 0.5) * nodesNum )), len(seq)-1) # 暂时设定为n
                noisy_dd = tll.fit(n=nodesNum, seq=seq, maxPartition=maxPartition, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=0)
            elif algo == 'EdgeDPCommonNeighborDistributionAnonHist_paper_L1': # compared algo
                tll = EdgeDPCommonNeighborDistributionAnonHist_paper_L1()
                noisy_dd = tll.fit(n=nodesNum, seq=seq, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=0)

            # error
            for Tindex in range(len(TList)):
                T = TList[Tindex]
                T = min(T, nodesNum-2)
                if queryKey == 0:
                    error = DistributionMetrics.normalizedKSTest(hist[T:], noisy_dd[T:])
                elif queryKey == 1:
                    error = DistributionMetrics.normalizedEMD_T(hist, noisy_dd, T) # 只有这个值有意义
                errorsList[algoKey][Tindex].append(error)


                histEMDError = DistributionMetrics.EMD(hist[T:], noisy_dd[T:])
                histEMDErrorsList[algoKey][Tindex].append(histEMDError)

                L1Error = DistributionMetrics.normalizedL1(hist[T:], noisy_dd[T:])
                L1ErrorsList[algoKey][Tindex].append(L1Error)
                
                KSError = DistributionMetrics.normalizedKSTest(hist[T:], noisy_dd[T:])
                KSErrorsList[algoKey][Tindex].append(KSError)

        print("\n")

    return TList, errorsList, histEMDErrorsList, L1ErrorsList, KSErrorsList


caller(0, 1)

for dataKey in [6]:
# for dataKey in DATASET_LIST:
    for queryKey in range(1, 2):
        epsList, errorsList, histEMDErrorsList, L1ErrorsList, KSErrorsList = caller(dataKey=dataKey, queryKey=queryKey)
        print('dataKey={}, queryKey={} :'.format(dataKey, queryKey))#, errorsList, L1ErrorsList)

    
        import pickle
        file = open('output/010-01-3/data{}_query{}.pickle'.format(dataKey, queryKey), 'wb')
        
        pickle.dump({'x': epsList, 'errorsList': errorsList, 'histEMDErrorsList': histEMDErrorsList, 'L1ErrorsList': L1ErrorsList, 'KSErrorsList': KSErrorsList}, file)
        file.close()

