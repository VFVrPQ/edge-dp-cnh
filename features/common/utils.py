IS_CUR_FILE_RUN = (__name__ == '__main__') # 当前文件运行

import os
import sys
import pickle
import networkx as nx
import numpy as np

sys.path.append('../../')
from features.common.constants import DATA_NAMES
from features.DP.models.edgeDP import GraphStatCommonNeighbors

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

# 用于区分时当前文件测试运行，还是项目根目录运行
dataDir = "../../datasets/" # expes/2002-2.../目录运行
if IS_CUR_FILE_RUN: # 当前文件运行
    dataDir = '../../datasets/'

# 根据数据集得到真实的hist
def getTrueHist(dataKey):
    datafile = dataDir + DATA_NAMES[dataKey]
    datafileTxt = datafile+".txt"
    datafileMap = datafile + '-map.txt'
    if not os.path.isfile(datafileMap):
        translate(datafileTxt, datafileMap)
    else:
        print(datafileMap, 'file exists')

    # 是否缓存了hist
    histCacheFile = dataDir + '/cache/' + DATA_NAMES[dataKey] + '-true-hist.pickle'
    if not os.path.isfile(histCacheFile):
        print('no cache file', histCacheFile)
        G = nx.read_edgelist(datafileMap, nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))

        gs = GraphStatCommonNeighbors(G)
        nodesNum, edgesNum, MCN = len(G.nodes()), len(G.edges()), gs.maxA #assume nodesNum is given
        hist = gs.hist

        file = open(histCacheFile, 'wb')
        pickle.dump({'hist': hist, 'nodesNum': nodesNum, 'edgesNum': edgesNum, 'MCN': MCN}, file)
        file.close()
    else:
        file = open(histCacheFile, 'rb')
        data = pickle.load(file)
        file.close()
        hist, nodesNum, edgesNum, MCN = data['hist'], data['nodesNum'], data['edgesNum'], data['MCN']
    return {
        'hist': hist,
        'nodesNum': nodesNum,
        'edgesNum': edgesNum,
        'MCN': MCN,
    }

# list去掉最大的和最小的
def refineList(mylist, deletedNum=2):
    ''' 去除最大的2个和最小的2个
    '''
    sorted_list = list(sorted(mylist))
    new_list = []
    for i in range(deletedNum, len(sorted_list)-deletedNum):
        new_list.append(sorted_list[i])
    return new_list

# 每个元素乘上数值
def adaptorList(elist, scale):
    return np.array(elist) * scale

def hist2seq(hist):
    seq = []
    for i in range(len(hist)):
        if hist[i] > 0:
            seq.extend([i]*np.int64(hist[i]))
    return seq

# 转化maxSeqLength个
def hist2seqLimit(hist, maxSeqLength):
    seq = []
    left = maxSeqLength
    for i in range(len(hist)-1, -1, -1):
        if hist[i] > 0:
            pickNum = min(left, hist[i])
            seq.extend([i]*np.int64(pickNum))
            left -= pickNum
            if left == 0: break
    return seq

# 抽样点，用于画图
def sampleList(seq):
    # 倍数
    j, bs = 1, 1.3
    xList = []
    while (j < len(seq)):
        xList.append(int(j))
        j = j * bs
    xList = sorted(list(set(xList))) # 是按字母而不是数字大小排序的
    xList.append(len(seq))

    yList = []
    for x in xList:
        yList.append(seq[x-1]) #这边是减1
    return xList, yList

if __name__ == '__main__':
    for i in [11, 2, 4, 5, 7, 12]:
        getTrueHist(i)