import os
import math
import subprocess
import networkx as nx
from utils import is_number, extract_lines_with_two_numbers, nearest_power_of_2
from utils import remove_nodes_with_min_degree, generateAGraphFromEdges
from utils import readFile, writeFile, getEdgesFromFile
from utils import GraphStatCommonNeighbors

import sys
sys.path.append('../../../')
from features.common.metrics import DistributionMetrics
from features.common.constants import DATA_NAMES, DATASET_LIST

def generateEdges(nodeNum, intNumber, filePath):
  """
  根据给定的节点数和边数生成边集，并将边集数据保存到指定文件中。
  
  参数：
      nodeNum：节点数
      intNumber：边数
      filePath：保存边集数据的文件路径
      
  返回：
      edgeData：生成的边集数据
      
  示例：
      nodeNum = 10
      intNumber = 20
      filePath = "data/edges.txt"
      edgeData = generateEdges(nodeNum, intNumber, filePath)
  """
  if os.path.exists(filePath):
      edgeData = readFile(filePath)
      #  print('exists data', edgeData)
  else:
    command = [
      "./lib/generator_omp",
      "{}".format(nodeNum),
      "-e {}".format(intNumber)
    ]
    # print(' '.join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdoutStr = result.stdout.decode('utf-8')  # 转换标准输出为字符串
    # stderrStr = result.stderr.decode('utf-8')  # 转换标准错误输出为字符串
    edgeData = extract_lines_with_two_numbers(stdoutStr.split('\n'), '\t')
    writeFile(filePath, edgeData)
    # print('not exists data', data)
  return edgeData

def generateAGraph(nodeNum, intNumber, index=None, originalNodeNum=None, datasetName=None):
    """
    根据给定的节点数和边数生成一个图，并进行节点的重命名和统计。
    
    参数：
        nodeNum：节点数
        intNumber：边数
        index：索引，用于区分不同的图实例，可选，默认为None
        originalNodeNum：原始节点数，用于删除节点后的重命名，可选，默认为None
        datasetName：数据集名称，用于保存生成的图和数据，可选，默认为None
        
    返回：
        GraphStatCommonNeighbors(G)：根据节点的共同邻居统计信息生成的结果
        
    示例：
        nodeNum = 10
        intNumber = 20
        index = 1
        originalNodeNum = 8
        datasetName = "graph_data"
        result = generateAGraph(nodeNum, intNumber, index, originalNodeNum, datasetName)
    """
    if not os.path.exists('./data'): os.mkdir('./data')
    if not os.path.exists('./data/{}'.format(datasetName)): os.mkdir('./data/{}'.format(datasetName))
    filePath = "./data/{}/{}-{}-{}".format(datasetName, nodeNum, intNumber, index) if is_number(index) else "./data/{}/{}-{}".format(datasetName, nodeNum, intNumber)
    print('[generateAGraph] filePath={}'.format(filePath))

    edgeData = generateEdges(nodeNum, intNumber, filePath)
    G = generateAGraphFromEdges(edgeData, nodes=[i for i in range(2**nodeNum)])
    if originalNodeNum != None:
      G = remove_nodes_with_min_degree(G, len(G.nodes()) - originalNodeNum)
      # 重新命名节点标签
      G = nx.relabel_nodes(G, {node: i for i, node in enumerate(list(G.nodes))})
    return GraphStatCommonNeighbors(G)


def runEachCase(datasetName, gs, index):
  """
    运行每个测试用例的函数

    参数：
    nodeNum (int) -- 节点数
    intNumber (int) -- 边数

    返回：
    无返回值
  """
  # [舍弃]使用生成图生成一个synthetic graph作为original graph
  # gs = generateAGraph(nodeNum, intNumber)
  # 计算oringal graph的nodeNum, intNumber，并且algorithm to generated a synthetic graph
  # 转换为2^nodeNum
  newNodeNum = nearest_power_of_2(gs.nodesNum)
  # 每个节点有多少度，应该是看扩展后的
  newIntNumber = math.ceil(gs.edgesNum * 2 / (2**newNodeNum))
  print('[runEachCase] newNodeNum={}, newIntNumber={}'.format(newNodeNum, newIntNumber))
  synGs = generateAGraph(newNodeNum, newIntNumber, index=index, originalNodeNum=gs.nodesNum, datasetName=datasetName)

  # 计算EMD of common neighbor count sequence
  return DistributionMetrics.normalizedEMD(gs.hist, synGs.hist)

CASE = 11
folderName = './result'
def main(dataName, CASE):
  # 使用真实数据作为original graph
  datasetNameMap = datasetName + '-map.txt'
  print('[main] datasetNameMap={}'.format(datasetNameMap))
  edges = getEdgesFromFile('../../2001.ladder/Datasets/' + datasetNameMap)
  gs = GraphStatCommonNeighbors(generateAGraphFromEdges(edges))

  # 断点续存
  if not os.path.exists(folderName): os.mkdir(folderName)
  filePath = '{}/{}'.format(folderName, datasetName)
  resultList = []
  if os.path.exists(filePath):
    resultList = readFile(filePath)
  for index in range(len(resultList), CASE):
    print('[main] dataName={}, index={}'.format(dataName, index))
    res = runEachCase(dataName, gs, index)
    resultList.append(res)
    writeFile(filePath)
  return resultList, CASE

# for i in range(len(DATASET_LIST)):
for i in [1, 5, 2, 3, 4]:
  datasetName = DATA_NAMES[DATASET_LIST[i]]
  resultList, _ = main(datasetName, CASE=CASE)
  print('[final] datasetName={}, resultList={}'.format(datasetName, resultList))