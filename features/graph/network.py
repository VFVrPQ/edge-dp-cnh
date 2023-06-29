import numpy as np
import networkx as nx
import community as community_louvain

def louvain(V, E):
    '''louvain得到labels
    '''
    nxG = nx.Graph()
    nxG.add_nodes_from(V)
    nxG.add_edges_from(E)

    partition = community_louvain.best_partition(nxG)
    partition = community_louvain.best_partition(nxG)
    npLabels = np.array(list(partition.values())) # 注意louvain算法本身是带随机的，以后尽量不要再重复运行
    

    print(len(nxG.edges()))
    print(nx.is_connected(nxG)) # 图是连通的
    return npLabels