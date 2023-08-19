import numpy as np
from utils import readFile, scientific_notation

import sys
sys.path.append('../../../')
from features.common.constants import DATA_NAMES, DATASET_LIST, DATA_NODES_NUMBER

CASE = 11
folderName = './result'
# for i in [1, 5, 2, 3, 4, 0]:
for i in [0, 1, 2, 3, 4, 5]:
  datasetName = DATA_NAMES[DATASET_LIST[i]]
  nodesNum = DATA_NODES_NUMBER[DATASET_LIST[i]]
  resultList = readFile('{}/{}'.format(folderName, datasetName))
  # print('datasetName={}'.format(datasetName), len(resultList), np.mean(resultList)*nodesNum*(nodesNum-1)//2, np.std(resultList)*nodesNum*(nodesNum-1)//2)
  # print('datasetName={}, resultList={}\n'.format(datasetName, resultList))
  print('{} & ${} \pm {}$ \\\\ \hline'.format(datasetName, scientific_notation(np.mean(resultList)*nodesNum*(nodesNum-1)//2), scientific_notation(np.std(resultList)*nodesNum*(nodesNum-1)//2)))