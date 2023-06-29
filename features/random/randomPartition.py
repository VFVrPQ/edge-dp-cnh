import random 
import numpy as np

def randomPartition(n, k):
    '''
        n: n个点
        k: k个clusters
        returns labels: 返回node i分配到的cluster编号
    '''
    labels = np.zeros(n, dtype=np.int64)
    for i in range(n):
        labels[i] = random.randint(0, k-1) # [low, high]
    return labels 