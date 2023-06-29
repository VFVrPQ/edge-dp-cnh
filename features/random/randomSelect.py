import numpy as np

def randomSelect2(d, k, l, r):
    '''随机均匀地选择k个点，每个点d维，范围为[l, r]
        d: number of dimensions 
        r: dataset range
        k: number of clusters
    '''
    randomCentroids = np.random.uniform(low=l, high=r, size=(k, d))
    return randomCentroids

def sampleProbList(probList):
    '''从列表中抽样
    '''
    #to-do: 可以用embedding中的方法加速
    normalizedProbList = probList / sum(probList)
    r = np.random.uniform(0, 1)
    s = 0
    for i in range(len(probList)):
        s += normalizedProbList[i]
        if s >= r:
            return i
    return len(probList)-1