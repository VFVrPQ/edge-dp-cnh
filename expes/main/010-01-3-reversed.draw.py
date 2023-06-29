import sys
import numpy as np

sys.path.append('../../')
from features.common.constants import DATA_NAMES, DATASET_LIST, MARKER_LIST, COLOR_LIST
from features.common.utils import refineList, getTrueHist, adaptorList

def refineList(mylist, deletedNum=2):
    ''' 去除最大的2个和最小的2个
    '''
    sorted_list = list(sorted(mylist))
    new_list = []
    for i in range(deletedNum, len(sorted_list)-deletedNum):
        new_list.append(sorted_list[i])
    return new_list


def drawResults(dataKey, errorKey=0, DRAW='curve'):
    deletedNum = 3
    errorList = ['errorsList', 'L1ErrorsList', 'KSErrorsList', 'histEMDErrorsList']
    errorNames = ['EMD', 'L1', 'KS', 'histEMD']
    queryKey = 1
    import pickle
    file = open('output/010-01-3-reversed/data{}_query{}.pickle'.format(dataKey, queryKey), 'rb')
    data = pickle.load(file)
    file.close()
    TList, errorsList = data['x'], data[errorList[errorKey]]

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
    
    if DRAW == 'curve':
        plt.clf() # 清空画布
        # 所有算法，依次为AnonHist、threeStages-SBA、threeStages-OBA、LMCHTruncated
        ccList = [0, 1, 2, 4] # 取对应的名称
        for i in [0, 2, 3]: # 枚举AnonHist、threeStages-OBA、LMCHTruncated
            y, yerr = [], []
            for j in range(len(errorsList[i])): # 枚举epsilon
                tempList = refineList(errorsList[i][j], deletedNum=deletedNum) # CASE个值
                mymean, mystd = np.mean(tempList), np.std(tempList) # 
                y.append(mymean)
                yerr.append(mystd)
            plt.plot(TList, y, label=nameList[ccList[i]], marker=MARKER_LIST[ccList[i]], c=COLOR_LIST[ccList[i]])
        plt.xticks([i*(200/5) for i in range(6)], size=15) # 指定横坐标
        plt.yticks(size=15) # 指定纵坐标
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0)) # 坐标轴设置为科学计数法表示
        plt.gca().yaxis.get_offset_text().set(size=15)  # 左上角1e6的字体大小
        plt.xlabel('$T$', fontsize=15)
        plt.ylabel(errorNames[errorKey], fontsize=15)
        # plt.ylim(ymin=0, ymax=min(ymax+ymax*0.1, 1)) # y轴最大刻度是2
        plt.legend(fontsize=14)
        plt.gca().invert_xaxis()  #翻转x轴
        plt.tight_layout()
        plt.savefig('output/010-01-3-reversed/data{}_query{}_error{}.pdf'.format(dataKey, queryKey, errorKey))
        plt.show()
    elif DRAW == 'bar':
        pass

    print(DATA_NAMES[dataKey])
    for i in range(len(errorsList[0])):
        print('T={}'.format(TList[i]))
        for j in range(len(errorsList)):
            tempList = refineList(errorsList[j][i], deletedNum=deletedNum) # CASE个值
            print(nameList[j], ':', np.mean(errorsList[j][i]), '$\pm$', np.std(errorsList[j][i]))
            
        print()

for i in DATASET_LIST:
# for i in [4]:
    drawResults(i, DRAW='curve')
