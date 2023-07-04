import numpy as np
import sys
sys.path.append('../../')
from features.common.constants import DATA_NAMES, DATASET_LIST, COLOR_LIST, MARKER_LIST
from features.common.utils import refineList, getTrueHist, adaptorList

dataNames = DATA_NAMES


def drawResults(dataKey, errorKey=0, DRAW='curve'):
    deletedNum = 3
    errorList = ['errorsList', 'L1ErrorsList', 'KSErrorsList', 'histEMDErrorsList']
    errorNames = ['EMD', 'L1', 'KS', 'histEMD']
    # dataKey, queryKey = 1, 1
    queryKey = 1
    import pickle
    file = open('output/010-01-3/data{}_query{}.pickle'.format(dataKey, queryKey), 'rb')
    data = pickle.load(file)
    file.close()
    # print(data)
    TList, errorsList = data['x'], data[errorList[errorKey]]
    from matplotlib import pyplot as plt
    nameList = [
        'AnonHist', # 'EdgeDPCommonNeighborDistributionAnonHist',
        # # 'threeStages', # 'threeStagesLL2_r_then_tau_new',
        'threeStages-SBA', # 'threeStagesLL2_r_then_tau_epsilon2',
        'threeStages-OBA', # 'threeStagesLL2_r_then_tau_epsilon2_new',
        # 'LMCH',
        'LMCHTruncated'
    ]

    if DRAW == 'curve':
        plt.clf() # 清空画布
        markerList = ['o', '^', 'x', 's', '+', 'p', '1', 'D']
        cList = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        for i in [0, 2, 3]: # 枚举算法
            y, yerr = [], []
            for j in range(len(errorsList[i])): # 枚举epsilon
                tempList = refineList(errorsList[i][j], deletedNum=deletedNum) # CASE个值
                mymean, mystd = np.mean(tempList), np.std(tempList) # 
                y.append(mymean)
                yerr.append(mystd)
            plt.plot(TList, y, label=nameList[i], marker=markerList[i], c=cList[i])
        plt.xticks([i*(200/5) for i in range(6)], size=15) # 指定横坐标
        plt.yticks(size=15) # 指定横坐标
        plt.xlabel('$T$', fontsize=15)
        plt.ylabel(errorNames[errorKey], fontsize=15)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('output/010-01-3/data{}_query{}_error{}.pdf'.format(dataKey, queryKey, errorKey))
        plt.show()

    print(dataNames[dataKey])
    for i in range(len(errorsList[0])):
        print('T={}'.format(TList[i]))
        for j in range(len(errorsList)):
            tempList = refineList(errorsList[j][i], deletedNum=deletedNum) # CASE个值
            print(nameList[j], ':', np.mean(errorsList[j][i]), '$\pm$', np.std(errorsList[j][i]))
            
        print()

for i in DATASET_LIST:
    drawResults(i, DRAW='curve')
