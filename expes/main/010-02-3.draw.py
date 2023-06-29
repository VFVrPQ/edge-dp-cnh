import numpy as np
import sys

sys.path.append('../../')
from features.common.constants import DATA_NAMES, DATASET_LIST, COLOR_LIST, MARKER_LIST
from features.common.utils import refineList, getTrueHist, adaptorList

def drawResults(dataKey, T=0, errorKey=0, DRAW=True):
    deletedNum = 3
    errorList = ['errorsList', 'L1ErrorsList', 'KSErrorsList', 'histEMDErrorsList']
    errorNames = ['EMD', 'L1', 'KS', 'histEMD']
    queryKey = 1
    import pickle
    file = open('output/010-02-3-T{}/data{}_query{}.pickle'.format(T, dataKey, queryKey), 'rb')
    data = pickle.load(file)
    file.close()
    # errorsList: 算法, epsilon, CASE个值
    epsList, errorsList = data['x'], data[errorList[errorKey]]

    # 放大到histogram EMD
    trueHistObj = getTrueHist(dataKey=dataKey)
    nodeNum = trueHistObj['nodesNum']
    errorsList = adaptorList(errorsList, (nodeNum * (nodeNum - 1) // 2))

    from matplotlib import pyplot as plt
    nameList = [
        'AnonHist', # 'EdgeDPCommonNeighborDistributionAnonHist',
        'threeStages-SBA', # 'threeStagesLL2_r_then_tau_epsilon2',
        'threeStages-OBA', # 'threeStagesLL2_r_then_tau_epsilon2_new',
        'LMCH',
        'LMCHTruncated'
    ]
    print('errorsList', len(errorsList));
    if DRAW == True:
        plt.clf() # 清空画布
        ymax = 0
         # 所有算法，依次为AnonHist、threeStages-SBA、threeStages-OBA、LMCH、LMCHTruncated
        for i in [0, 2, 4]: # 枚举算法
            y = []
            for j in range(len(errorsList[i])): # 枚举epsilon
                tempList = refineList(errorsList[i][j], deletedNum=deletedNum) # CASE个值
                mymean, mystd = np.mean(tempList), np.std(tempList) # 
                y.append(mymean)
                if nameList[i] != 'LMS':
                    ymax = max(ymax, min(1, mymean))
            plt.semilogy(epsList, y, label=nameList[i], marker=MARKER_LIST[i], c=COLOR_LIST[i])
        plt.xticks([i/10 for i in range(1, 20, 2)], size=15) # 指定横坐标
        plt.yticks(size=15) 
        plt.xlabel('$\epsilon$', fontsize=15)
        plt.ylabel(errorNames[errorKey], fontsize=15)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('output/010-02-3-T{}/data{}_query{}_error{}.pdf'.format(T, dataKey, queryKey, errorKey))
        plt.show()

    print(DATA_NAMES[dataKey], 'T={}'.format(T))
    for i in range(len(errorsList[0])):
        if (epsList[i] != 0.5 and epsList[i] != 1.0 and epsList[i] != 1.5): continue
        print('epsilon={}'.format(epsList[i]))
        for j in range(len(errorsList)):
            tempList = refineList(errorsList[j][i], deletedNum=deletedNum) # CASE个值
            print('{} : {:.3f} $\pm {:.3f}'.format(nameList[j], np.mean(errorsList[j][i]), np.std(errorsList[j][i])))
        print()

for i in DATASET_LIST: # dataset
    for T in [_*40 for _ in range(0, 1)]: # TList
        drawResults(i, T=T, DRAW=True)
