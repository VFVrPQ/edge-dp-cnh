import numpy as np
dataNames = ["toydata", "facebook_combined", "wiki-Vote", 
        "CA-GrQc",
        "email-Enron",  
        "cit-HepTh",  # memory error
        'deezer_europe_edges',
        'musae_git_edges',
        'CA-HepPh',
        'CA-CondMat',
        'email-Eu-core',
        'Cit-HepPh',
        'musae_chameleon_edges',
        'musae_squirrel_edges',
        'musae_PTBR_edges'
        # "com-dblp.ungraph"
        ]
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
        'LMCH',
        'LMCHTruncated'
    ]

    if DRAW == 'curve':
        plt.clf() # 清空画布
        # markerList = ['o', '^', 'x', '+', 's', 'p', '1', 'D']
        markerList = ['o', '^', 'x', 's', '+', 'p', '1', 'D']
        cList = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        ccList = [0, 1, 2, 4]
        for i in range(len(errorsList)): # 枚举算法
            y, yerr = [], []
            for j in range(len(errorsList[i])): # 枚举epsilon
                tempList = refineList(errorsList[i][j], deletedNum=deletedNum) # CASE个值
                mymean, mystd = np.mean(tempList), np.std(tempList) # 
                y.append(mymean)
                yerr.append(mystd)
            plt.plot(TList, y, label=nameList[ccList[i]], marker=markerList[ccList[i]], c=cList[ccList[i]])
        plt.xticks([i*(200/5) for i in range(6)], size=15) # 指定横坐标
        plt.yticks(size=15) # 指定横坐标
        plt.xlabel('$T$', fontsize=15)
        plt.ylabel(errorNames[errorKey], fontsize=15)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('output/010-01-3/data{}_query{}_error{}.pdf'.format(dataKey, queryKey, errorKey))
        plt.show()
    elif DRAW == 'bar':
        markerList = ['o', '^', 'x', 's', '+', 'p', '1', 'D']
        cList = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

        barWidth = 1 / (len(errorsList) + 1)
        # for i in range(len(errorsList)): # 枚举算法
        for i in [1, 2, 4]: # 枚举算法
            y, yerr = [], []
            for j in range(len(errorsList[i])): # 枚举epsilon
                tempList = refineList(errorsList[i][j], deletedNum=deletedNum) # CASE个值
                mymean, mystd = np.mean(tempList), np.std(tempList) # 
                y.append(mymean)
                yerr.append(mystd)

            br = [j + barWidth * i for j in range(len(errorsList[i]))]
            plt.bar(br, y, label=nameList[i], c=cList[i], width=barWidth, yerr=yerr, log=False)
        plt.xticks([j + barWidth * len(errorsList) / 2 for j in range(len(errorsList[i]))], TList, size=15) # 指定横坐标
        plt.yticks(size=15) # 指定横坐标
        plt.xlabel('$T$', fontsize=15)
        plt.ylabel(errorNames[errorKey], fontsize=15)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig('output/010-01-3/data{}_query{}_error{}.pdf'.format(dataKey, queryKey, errorKey))
        plt.show()        
        pass

    print(dataNames[dataKey])
    for i in range(len(errorsList[0])):
        print('T={}'.format(TList[i]))
        for j in range(len(errorsList)):
            tempList = refineList(errorsList[j][i], deletedNum=deletedNum) # CASE个值
            print(nameList[j], ':', np.mean(errorsList[j][i]), '$\pm$', np.std(errorsList[j][i]))
            
        print()

for i in [2, 4, 5, 7, 12]:
    drawResults(i, DRAW='curve')
