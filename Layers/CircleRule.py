'''
Created on 27.09.2019

@author: florian
'''

import numpy as np
import networkx as nx

import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from networkx.algorithms.cycles import cycle_basis

path = "/home/florian/Dokumente/Databases/GraphData/DS_all/"
#path = "/home/seiffart/Documents/Syncronizing/Forschung/Forschung/EigeneForschung/RuleBasedNeuralNetworks/DS_all/"
#path = "F:/EigeneDokumente/Forschung/EigeneForschung/RuleBasedNeuralNetworks/DS_all/"
db = "MUTAG"

"""
Data preprocessing: get the graphs
"""
graph_data = gdtgl.graph_data_to_graph_list(path, db)
graph_list, graph_labels, graph_attributes = graph_data
graph_number = len(graph_list)

G = graph_list[3]
cycle_list = nx.minimum_cycle_basis(G)
print(cycle_list)

W = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

for i in range(0, W.size):
    for j in range(0, W[0].size):
        for t in cycle_list:
            if i in t and j in t:
                W[i][j] = len(t)

print(W)
gdtgl.draw_graph(G)

