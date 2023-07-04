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
    else:
        print("file exists")


    queryTypeList = ["KSTest", 'EMD']
    queryType = queryTypeList[queryKey]
    print('queryType = {}'.format(queryType))

    algoNames = [
        'EdgeDPCommonNeighborDistributionAnonHist_paper_L1',
        'threeStagesLL2_r_then_tau_epsilon2',
        'threeStagesLL2_r_then_tau_epsilon2_new',
        'LMCH',
        'LMCHTruncated'
    ]
    # algoNames = ['LMCHTruncated']
    epsList = []
    epsList.extend([i/10 for i in range(1, 20, 1)])
    repeats = 30
    T = 0
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

        for epsilon in epsList:
            T = 0
            errors = []
            histEMDErrors = []
            L1Errors = []
            KSErrors = []

            for i in range(repeats):
                if algo == 'LMCH':
                    lc = EdgeDPCommonNeighborDistributionLaplaceCumulateHistogram()
                    noisy_dd = lc.fit(n=nodesNum, b=hist, sens=2*(nodesNum-2), epsilon=epsilon, T=T)
                elif algo == 'LMCHTruncated':
                    lc = EdgeDPCommonNeighborDistributionLaplaceCumulateHistogramTruncated()
                    noisy_dd = lc.fit(n=nodesNum, b=hist, sens=2*(nodesNum-2), epsilon=epsilon, T=T)
                # SBA
                elif algo == 'threeStagesLL2_r_then_tau_epsilon2':
                    tll = EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022(getChooseRTauParam='r_then_tau_epsilon2')
                    noisy_dd = tll.fit(n=nodesNum, seq=seq, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=T)
                # threeStages-OBA
                elif algo == 'threeStagesLL2_r_then_tau_epsilon2_new':
                    tll = EdgeDPCommonNeighborDistributionSequenceThreeStagesLL20022(getChooseRTauParam='r_then_tau_epsilon2_new')
                    noisy_dd = tll.fit(n=nodesNum, seq=seq, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=T)
                elif algo == 'EdgeDPCommonNeighborDistributionAnonHist_paper_L1': # compared algo
                    tll = EdgeDPCommonNeighborDistributionAnonHist_paper_L1()
                    noisy_dd = tll.fit(n=nodesNum, seq=seq, sens=2*(nodesNum-2), epsilon=epsilon, hist=hist, T=T)

                # error
                if queryKey == 0:
                    error = DistributionMetrics.normalizedKSTest(hist[T:], noisy_dd[T:])
                elif queryKey == 1:
                    error = DistributionMetrics.normalizedEMD(hist[T:], noisy_dd[T:])
                errors.append(error)


                histEMDError = DistributionMetrics.EMD(hist[T:], noisy_dd[T:])
                histEMDErrors.append(histEMDError)

                L1Error = DistributionMetrics.normalizedL1(hist[T:], noisy_dd[T:])
                L1Errors.append(L1Error)
                
                KSError = DistributionMetrics.normalizedKSTest(hist[T:], noisy_dd[T:])
                KSErrors.append(KSError)

                print('data = {}, algo = {}, epsilon = {}, T = {}: sum(hist[T:]) = {}, sum(noisy_hist[T:]) = {}, error = {}'.format(
                    dataName, algo, epsilon, T, np.sum(hist[T:]), np.sum(noisy_dd[T:]), error))
            print(epsilon, T, ' EMD :', np.mean(errors), ' $\pm$ ', np.std(errors), errors)
            print(epsilon, T, ' histEMD :', np.mean(histEMDErrors), ' $\pm$ ', np.std(histEMDErrors), histEMDErrors)
            print("\n")

            # 修改, 保存所有结果
            errorsList[algoKey].append(errors)
            histEMDErrorsList[algoKey].append(histEMDErrors)
            L1ErrorsList[algoKey].append(L1Errors)
            KSErrorsList[algoKey].append(KSErrors)
        print("\n")

    return epsList, errorsList, histEMDErrorsList, L1ErrorsList, KSErrorsList


caller(0, 1)


for dataKey in DATASET_LIST:
    for queryKey in range(1, 2):
        epsList, errorsList, histEMDErrorsList, L1ErrorsList, KSErrorsList = caller(dataKey=dataKey, queryKey=queryKey)
        print('dataKey={}, queryKey={} :'.format(dataKey, queryKey), errorsList, L1ErrorsList)

    
        import pickle
        file = open('output/010-02-2/data{}_query{}.pickle'.format(dataKey, queryKey), 'wb')
        
        pickle.dump({'x': epsList, 'errorsList': errorsList, 'histEMDErrorsList': histEMDErrorsList, 'L1ErrorsList': L1ErrorsList, 'KSErrorsList': KSErrorsList}, file)
        file.close()

