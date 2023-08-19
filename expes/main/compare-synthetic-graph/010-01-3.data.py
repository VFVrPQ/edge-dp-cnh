import numpy as np
from utils import readFile, refineList, scientific_notation

import sys
sys.path.append('../../../')
from features.common.constants import DATA_NODES_NUMBER, DATA_NAMES

def readFrom010_01_3(filePath, index):
    deletedNum = 3

    data = readFile(filePath)
    TList, errorsList = data['x'], data['errorsList']
    nameList = [
        'AnonHist', # 'EdgeDPCommonNeighborDistributionAnonHist',
        'threeStages-SBA', # 'threeStagesLL2_r_then_tau_epsilon2',
        'threeStages-OBA', # 'threeStagesLL2_r_then_tau_epsilon2_new',
        'LMCH',
        'LMCHTruncated'
    ]
    for i in range(len(errorsList[0])):
        if TList[i] != 0: continue
        print('\nT={}'.format(TList[i]))
        for j in range(len(errorsList)):
            tempList = refineList(errorsList[j][i], deletedNum=deletedNum) # CASE个值
            avg, std = np.mean(tempList) * (DATA_NODES_NUMBER[index] * (DATA_NODES_NUMBER[index] - 1 ) // 2), np.std(tempList) * (DATA_NODES_NUMBER[index] * (DATA_NODES_NUMBER[index] - 1 ) // 2)
            print(nameList[j], ':', avg, '$\pm$', std)

            if nameList[j] == 'AnonHist':
                return tempList


if __name__ == '__main__':
    DATASET_INDEX_MAP = {
        1: 11,
        2: 2,
        3: 4,
        4: 5,
        5: 7,
        6: 12
    }
    DIR = './result/010-01-3-reversed'


    for dataKey in [1, 2, 3, 4, 5, 6]:
      resultList = readFrom010_01_3('{}/data{}_query1.pickle'.format(DIR, DATASET_INDEX_MAP[dataKey]), dataKey)

      nodesNum = DATA_NODES_NUMBER[dataKey]
      print('{} : ${} \pm {}$'.format(DATA_NAMES[dataKey], scientific_notation(np.mean(resultList)*nodesNum*(nodesNum-1)//2), scientific_notation(np.std(resultList)*nodesNum*(nodesNum-1)//2)))