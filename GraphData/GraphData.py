from typing import List, Dict

import networkx as nx
import torch

import TrainTestData.TrainTestData as ttd
import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from GraphData import NodeLabeling, EdgeLabeling


class NodeLabels:
    def __init__(self):
        self.node_labels = None
        self.unique_node_labels = None
        self.db_unique_node_labels = None
        self.num_unique_node_labels = 0


class EdgeLabels:
    def __init__(self):
        self.edge_labels = None
        self.unique_edge_labels = None
        self.db_unique_edge_labels = None
        self.num_unique_edge_labels = 0


class GraphData:
    def __init__(self):
        self.graph_db_name = ''
        self.graphs = []
        self.inputs = []
        self.node_labels: Dict[str, NodeLabels] = {}
        self.edge_labels: Dict[str, EdgeLabels] = {}
        self.graph_labels = []
        self.one_hot_labels = []
        self.num_classes = 0
        # additonal parameters
        self.distance_list = None
        self.cycle_list = None
        self.max_nodes = 0

    def init_from_graph_db(self, path, graph_db_name, with_distances=False, with_cycles=False, relabel_nodes=False,
                           use_features=True, use_attributes=False, distances_path=None):
        distance_list = []
        cycle_list = []

        # CSL graphs
        if graph_db_name == 'CSL':
            from LoadData.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        else:
            # Define the graph data
            graph_data = gdtgl.graph_data_to_graph_list(path, graph_db_name, relabel_nodes=relabel_nodes)

        if with_distances:
            self.distance_list = []
        if with_cycles:
            self.cycle_list = []
        self.inputs, self.one_hot_labels, graph_data, self.distance_list = ttd.data_from_graph_db(graph_data,
                                                                                                  graph_db_name,
                                                                                                  self.cycle_list,
                                                                                                  one_hot_encode_labels=True,
                                                                                                  use_features=use_features,
                                                                                                  use_attributes=use_attributes,
                                                                                                  with_distances=with_distances,
                                                                                                  distances_path=distances_path)
        self.graphs = graph_data[0]
        self.graph_labels = graph_data[1]
        # num classes are unique labels
        self.num_classes = len(set(self.graph_labels))
        self.num_graphs = len(self.graphs)
        self.graph_db_name = graph_db_name

        # get graph with max number of nodes
        self.max_nodes = 0
        for g in self.graphs:
            self.max_nodes = max(self.max_nodes, g.number_of_nodes())

        # set primary node and edge labels
        self.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        self.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)

        return None

    def add_node_labels(self, node_labeling_name, max_label_num=-1, node_labeling_method=None, **kwargs) -> None:
        if node_labeling_method is not None:
            node_labeling = NodeLabels()
            node_labeling.node_labels, node_labeling.unique_node_labels, node_labeling.db_unique_node_labels = node_labeling_method(
                self.graphs, **kwargs)
            node_labeling.num_unique_node_labels = max(1, len(node_labeling.db_unique_node_labels))

            key = node_labeling_name
            if max_label_num is not None and max_label_num > 0:
                key = f'{node_labeling_name}_{max_label_num}'

            self.node_labels[key] = node_labeling
            self.relabel_most_frequent(self.node_labels[key], max_label_num)

    def add_edge_labels(self, edge_labeling_name, edge_labeling_method=None, **kwargs) -> None:
        if edge_labeling_method is not None:
            edge_labeling = EdgeLabels()
            edge_labeling.edge_labels, edge_labeling.unique_edge_labels, edge_labeling.db_unique_edge_labels = edge_labeling_method(
                self.graphs, **kwargs)
            edge_labeling.num_unique_edge_labels = max(1, len(edge_labeling.db_unique_edge_labels))
            self.edge_labels[edge_labeling_name] = edge_labeling

    def relabel_most_frequent(self, labels: NodeLabels, num_max_labels: int):
        # get the k most frequent node labels or relabel all
        if num_max_labels == -1:
            bound = len(labels.db_unique_node_labels)
        else:
            bound = min(num_max_labels, len(labels.db_unique_node_labels))
        most_frequent = sorted(labels.db_unique_node_labels, key=labels.db_unique_node_labels.get, reverse=True)[
                        :bound - 1]
        # relabel the node labels
        for i, _lab in enumerate(labels.node_labels):
            for j, lab in enumerate(_lab):
                if lab not in most_frequent:
                    labels.node_labels[i][j] = bound - 1
                else:
                    labels.node_labels[i][j] = most_frequent.index(lab)
        # set the new unique labels
        labels.num_unique_node_labels = bound
        db_unique = {}
        for i, l in enumerate(labels.node_labels):
            unique = {}
            for label in l:
                if label not in unique:
                    unique[label] = 1
                else:
                    unique[label] += 1
                if label not in db_unique:
                    db_unique[label] = 1
                else:
                    db_unique[label] += 1
            labels.unique_node_labels[i] = unique
        labels.db_unique_node_labels = db_unique
        pass



def zinc_to_networkx(train, validation, test, graph_db_name):
    graphs = GraphData()
    graphs.graph_db_name = graph_db_name
    graphs.edge_labels['primary'] = EdgeLabels()
    graphs.node_labels['primary'] = NodeLabels()
    graphs.node_labels['primary'].node_labels = []
    graphs.edge_labels['primary'].edge_labels = []
    graphs.graph_labels = []
    graphs.max_nodes = 0
    graphs.num_classes = 1
    graphs.num_graphs = len(train) + len(validation) + len(test)

    original_source = -1
    for data in [train, validation, test]:
        for i, graph in enumerate(data):
            # add new graph
            graphs.graphs.append(nx.Graph())
            graphs.edge_labels['primary'].edge_labels.append([])
            graphs.inputs.append(torch.ones(graph['x'].shape[0]).double())

            edges = graph['edge_index']
            # format edges to list of tuples
            edges = edges.T.tolist()
            # add edges to graph
            for i, edge in enumerate(edges):
                if edge[0] < edge[1]:
                    graphs.graphs[-1].add_edge(edge[0], edge[1])
                    graphs.edge_labels['primary'].edge_labels[-1].append(graph['edge_attr'][i].item())
            # add node labels
            graphs.node_labels['primary'].node_labels.append([x.item() for x in graph['x']])
            # add graph inputs using the values from graph['x'] and flatten the tensor
            graphs.inputs[-1] = graph['x'].flatten().double()

            graphs.edge_labels['primary'].edge_labels.append(graph['edge_attr'])
            graphs.node_labels['primary'].node_labels.append([graph['x'][node].item() for node in graphs.graphs[-1].nodes])
            graphs.graph_labels.append(graph['y'].item())
            graphs.max_nodes = max(graphs.max_nodes, len(graph['x']))



            pass
        pass
    graphs.one_hot_labels = graphs.graph_labels
    return graphs