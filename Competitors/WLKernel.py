#
from typing import List
import grakel
import numpy as np
from grakel import GraphKernel

from GraphData.GraphData import GraphData
from grakel.utils import graph_from_networkx

from grakel.datasets import fetch_dataset

from sklearn.model_selection import train_test_split


class WLKernel:
    def __init__(self, graph_data: GraphData, run_num: int, validation_num: int, training_data: List[int],
                 validate_data: List[int], test_data: List[int],
                 seed: int):
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.run_num = run_num
        self.validation_num = validation_num

    def Run(self):
        # create numpy vector from the graph data labels
        primary_node_labels = self.graph_data.node_labels['primary']
        primary_edge_labels = self.graph_data.edge_labels['primary']


        train_graphs = [self.graph_data.graphs[i] for i in self.training_data]
        val_graphs = [self.graph_data.graphs[i] for i in self.validate_data]
        test_graphs = [self.graph_data.graphs[i] for i in self.test_data]

        y_train = np.asarray([self.graph_data.graph_labels[i] for i in self.training_data])
        y_val = np.asarray([self.graph_data.graph_labels[i] for i in self.validate_data])
        y_test = np.asarray([self.graph_data.graph_labels[i] for i in self.test_data])

        grakel_train = grakel.graph_from_networkx(train_graphs, node_labels_tag='label')
        grakel_val = grakel.graph_from_networkx(val_graphs, node_labels_tag='label')
        grakel_test = grakel.graph_from_networkx(test_graphs, node_labels_tag='label')

        # create input for the kernel from the grakel graphs
        grakel_train_input = []
        for g in train_graphs:
            edge_set = set()
            edge_dict = {}
            edges = g.edges(data=True)
            for e in edges:
                label = 0
                if 'label' in e[2]:
                    label = e[2]['label']
                edge_dict[(e[0], e[1])] = label
                edge_set.add((e[0], e[1]))
                edge_set.add((e[1], e[0]))
            node_dict = {}
            for n in g.nodes(data=True):
                label = 0
                if 'label' in n[1]:
                    label = n[1]['label']
                node_dict[n[0]] = label
            grakel_train_input.append([edge_set, node_dict, edge_dict])





        gk = GraphKernel(kernel=[{"name": "weisfeiler_lehman", "n_iter": 5}, "subtree_wl"], Nystroem=20)
        K_train = gk.fit_transform(grakel_train)
        K_val = gk.transform(grakel_val)
        K_test = gk.transform(grakel_test)

        grakel.WeisfeilerLehman(n_iter=1).fit_transform(train_graphs)

        X = np.zeros(shape=(self.graph_data.num_graphs,
                            primary_node_labels.num_unique_node_labels + primary_edge_labels.num_unique_edge_labels))
        # fill the numpy vector with number of unique node and edge labels
        for i in range(0, self.graph_data.num_graphs):
            for j in range(0, primary_node_labels.num_unique_node_labels):
                if j in primary_node_labels.unique_node_labels[i]:
                    X[i, j] = primary_node_labels.unique_node_labels[i][j]
            for j in range(primary_node_labels.num_unique_node_labels,
                           primary_node_labels.num_unique_node_labels + primary_edge_labels.num_unique_edge_labels):
                if j - primary_node_labels.num_unique_node_labels in primary_edge_labels.unique_edge_labels[i]:
                    X[i, j] = primary_edge_labels.unique_edge_labels[i][j - primary_node_labels.num_unique_node_labels]

        Y = np.asarray(self.graph_data.graph_labels)
        # split the data in training, validation and test set
        X_train = X[self.training_data]
        Y_train = Y[self.training_data]
        X_val = X[self.validate_data]
        Y_val = Y[self.validate_data]
        X_test = X[self.test_data]
        Y_test = Y[self.test_data]

        for c_param in range(1, 9):
            wl = grakel.WeisfeilerLehman(n_iter=c_param)
