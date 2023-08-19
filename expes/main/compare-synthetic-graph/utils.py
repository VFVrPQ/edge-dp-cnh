import re
import time
import math
import random
import pickle
import numpy as np
import networkx as nx
from collections import defaultdict

def scientific_notation(num):
    x, y = "{:.2e}".format(num).split('e')
    result = "{} \\times 10^{}".format(x, int(y))
    return result

def is_number(variable):
    if isinstance(variable, (int, float)):
        return True
    elif isinstance(variable, str) and variable.isdigit():
        return True
    else:
        return False

def extract_lines_with_two_numbers(lines, sep='\t'):
    extracted_lines = []
    for line in lines:
        if re.match(r'\d+\s+\d+', line):
            extracted_lines.append([int(i) for i in line.split(sep)])
    return extracted_lines

def nearest_power_of_2(num):
    if num & (num - 1) == 0:
        return int(math.log2(num))
    
    return math.ceil(math.log2(num))


def generateAGraphFromEdges(edges, nodes=None):
  """
    根据给定的边集生成一个图，并可选择添加节点。
    
    参数：
        edges：边集，形式为[(node1, node2), (node2, node3), ...]
        nodes：节点集，形式为[node1, node2, ...]，默认为None
        
    返回：
        graph：生成的图对象
        
    示例：
        edges = [(1, 2), (2, 3), (3, 4)]
        nodes = [1, 2, 3, 4, 5]
        graph = generateAGraphFromEdges(edges, nodes)
  """
  print('[generateAGraphFromEdges]')
  # 创建一个空图
  graph = nx.Graph()
  # 添加边集到图中
  graph.add_edges_from(edges)
  # 删除重复边
  graph.remove_edges_from(nx.selfloop_edges(graph))
  # 重新命名节点标签
  graph = nx.relabel_nodes(graph, {node: i for i, node in enumerate(list(graph.nodes))})
  # 添加点的编号
  if nodes: graph.add_nodes_from(nodes)
  return graph

def remove_nodes_with_min_degree(G, x):
    '''这个函数的作用是删除图G中度数最小的x个节点及其边。函数接受两个参数：图G和要删除的节点数量x。函数会返回删除节点后的图G。
    '''
    print('[remove_nodes_with_min_degree] G.nodesNum={}, x={}', len(G.nodes()), x)
    # 找到度数最小的x个节点
    degrees = dict(G.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1])
    min_degree_nodes = [node for node, degree in sorted_degrees[:x]]

    # print('G.nodes()', G.nodes(), sorted_degrees)
    # 删除度数最小的x个节点及其边
    for node in min_degree_nodes:
        edges_to_remove = [(node, neighbor) for neighbor in G.neighbors(node)]
        G.remove_edges_from(edges_to_remove)
    G.remove_nodes_from(min_degree_nodes)
    return G

def remove_random_edges(graph, num_edges):
    # 获取图的边集合
    edges = list(graph.edges())
    
    # 如果需要删除的边数大于图中边的总数，则将需要删除的边数设置为图中边的总数
    num_edges = min(num_edges, len(edges))
    
    # 随机选择要删除的边
    edges_to_remove = random.sample(edges, num_edges)
    
    # 从图中删除选定的边
    graph.remove_edges_from(edges_to_remove)
    
    return graph

# 保存数据到文件
def writeFile(filePath, data):
    with open(filePath, 'wb') as file:
        pickle.dump(data, file)

# 从文件中读取数据
def readFile(filePath):
    with open(filePath, 'rb') as file:
        return pickle.load(file)

def refineList(mylist, deletedNum=2):
    ''' 去除最大的2个和最小的2个
    '''
    sorted_list = list(sorted(mylist))
    new_list = []
    for i in range(deletedNum, len(sorted_list)-deletedNum):
        new_list.append(sorted_list[i])
    return new_list

def getEdgesFromFile(filePath):
    """
        从文件中读取边的数据。

        Args:
            filePath (str): 文件路径。

        Returns:
            list: 包含边的列表，每个元素是一个包含两个数字的元组。
    """
    with open(filePath, 'r') as f:
        data = f.readlines()
        return extract_lines_with_two_numbers(data, sep=' ')


class GraphStatCommonNeighbors(object):
    '''mainly store aggregated statistics of G
        Parameters
        ----------
        G: networkx graph

        Returns
        -------
        nodesNum: 节点个数

        maxA: 最大度数

        A: 任意点对间公共邻居的集合

        References
        ----------
        .. https://github.com/DPGraph/DPGraph/blob/91a87c95b39e46f4c2e78ba0edff9d8d9bf732d5/util.py#L150

    '''
    def __init__(self, G):
        # degree number
        self.nodesNum, self.edgesNum = len(G.nodes()), len(G.edges())

        # a_ij: the number of common neighbors of i and j
        self.a = defaultdict(int)
        # 最大是n-2,hist统计
        self.hist = np.zeros(shape=(self.nodesNum-1,), dtype=np.int64)
        self.hist[0] = np.int64(self.nodesNum * (self.nodesNum - 1) / 2)

        self.initSparseA(G)
        print('[GraphStatCommonNeighbors] nodesNum={}, edgesNum={}'.format(self.nodesNum, self.edgesNum))

    def initSparseA(self, G):
        startTime = time.time()
        for u in range(len(G.nodes())):
            neList = [v for v in G[u]]
            # 邻居序列里的均是公共邻居
            for p in range(len(neList)):
                for q in range(p):
                    v, w = neList[p], neList[q]
                    if v>w: v, w = w, v
                    self.a[(v, w)] += 1
                    # 上述两个算同一个点对
                    self.hist[ self.a[(v, w)] ] += 1
                    self.hist[ self.a[(v, w)]-1 ] -= 1
        print('---------initSparseA: {} seconds--------------'.format(time.time() - startTime))

    def geta(self, i, j):
        '''公共邻居个数
        '''
        return self.a[i][j]

    # def get_common_neighbor_counts(self, graph):
    #     print('---------get_common_neighbor_counts begins--------------')
    #     startTime = time.time()
    #     common_neighbor_counts = defaultdict(int)
    #     nodes = list(graph.nodes())

    #     for i in range(len(nodes)):
    #         for j in range(i+1, len(nodes)):
    #             node1 = nodes[i]
    #             node2 = nodes[j]
    #             count = len(list(nx.common_neighbors(graph, node1, node2)))
    #             common_neighbor_counts[(node1, node2)] = count

    #     print('---------get_common_neighbor_counts: {} seconds--------------'.format(time.time() - startTime))
    #     return common_neighbor_counts
