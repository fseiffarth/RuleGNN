import os
import sys
from typing import Dict, List

import networkx as nx
import torch
import torch_geometric.data
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import ZINC

import TrainTestData.TrainTestData as ttd
import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from GraphData import NodeLabeling, EdgeLabeling
from GraphData.Distances.load_distances import load_distances
from utils.GraphLabels import NodeLabels, EdgeLabels, Properties
from utils.utils import load_graphs


class GraphData:
    def __init__(self):
        self.graph_db_name = ''
        self.graphs = []
        self.inputs = []
        self.node_labels: Dict[str, NodeLabels] = {}
        self.edge_labels: Dict[str, EdgeLabels] = {}
        self.properties: Dict[str, Properties] = {}
        self.graph_labels = []
        self.one_hot_labels = []
        self.num_classes = 0
        self.max_nodes = 0

    def init_from_graph_db(self, path, graph_db_name, with_distances=False, with_cycles=False, relabel_nodes=False,
                           use_features=True, use_attributes=False, distances_path=None):
        distance_list = []
        cycle_list = []

        # CSL graphs
        if graph_db_name == 'CSL':
            from GraphBenchmarks.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        else:
            # Define the graph data
            graph_data = gdtgl.graph_data_to_graph_list(path, graph_db_name, relabel_nodes=relabel_nodes)

        if with_distances:
            self.distance_list = []
        if with_cycles:
            self.cycle_list = []
        self.inputs, self.one_hot_labels, graph_data = ttd.data_from_graph_db(graph_data=graph_data,
                                                                              graph_db_name=graph_db_name,
                                                                              one_hot_encode_labels=True,
                                                                              use_features=use_features,
                                                                              use_attributes=use_attributes, )
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

    def load_from_benchmark(self, db_name, path, use_features=True):
        self.graph_db_name = db_name
        self.graphs, self.graph_labels = load_graphs(path, db_name)
        self.num_graphs = len(self.graphs)
        try:
            self.num_classes = len(set(self.graph_labels))
        except:
            self.num_classes = len(self.graph_labels[0])
        self.max_nodes = 0
        for g in self.graphs:
            self.max_nodes = max(self.max_nodes, g.number_of_nodes())

        self.one_hot_labels = torch.zeros(self.num_graphs, self.num_classes)

        for i, label in enumerate(self.graph_labels):
            if type(label) == int:
                self.one_hot_labels[i][label] = 1
            elif type(label) == list:
                self.one_hot_labels[i] = torch.tensor(label)

        self.inputs = []
        ## add node labels
        for graph in self.graphs:
            self.inputs.append(torch.ones(graph.number_of_nodes()).float())
            if use_features:
                for node in graph.nodes(data=True):
                    self.inputs[-1][node[0]] = node[1]['label'][0]

        self.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        self.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)

        # normalize the graph inputs, i.e. to have values between 0 and 1
        if use_features:
            # get the number of different node labels
            num_node_labels = self.node_labels['primary'].num_unique_node_labels
            for i, graph in enumerate(self.inputs):
                self.inputs[i] /= num_node_labels

        return None


def get_graph_data(db_name, data_path, use_features=None, use_attributes=None, with_distances=True):
    """
    Load the graph data by name.
    :param db_name: str - name of the graph database
    :param data_path: str - path to the data

    """
    # load the graph data
    if db_name == 'CSL':
        from GraphBenchmarks.csl import CSL
        csl = CSL()
        graph_data = csl.get_graphs(with_distances=False)
    elif db_name == 'ZINC':
        zinc_train = ZINC(root="../../ZINC/", subset=True, split='train')
        zinc_val = ZINC(root="../../ZINC/", subset=True, split='val')
        zinc_test = ZINC(root="../../ZINC/", subset=True, split='test')
        graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
    elif ('LongRings' in db_name) or ('EvenOddRings' in db_name) or ('SnowflakesCount' in db_name) or (
            'Snowflakes' in db_name):
        graph_data = GraphData()
        # add db_name and raw to the data path
        data_path = data_path + db_name + "/raw/"
        graph_data.load_from_benchmark(db_name, data_path, use_features)
    else:
        graph_data = GraphData()
        graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                      relabel_nodes=True, use_features=use_features, use_attributes=use_attributes)
    return graph_data


class BenchmarkDatasets(InMemoryDataset):
    def __init__(self, root: str, name: str, graph_data: GraphData):
        self.graph_data = graph_data
        self.name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}_Edges.txt', f'{self.name}_Nodes.txt', f'{self.name}_Labels.txt']

    @property
    def processed_file_names(self):
        return [f'data.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        num_node_labels = self.graph_data.node_labels['primary'].num_unique_node_labels
        for i, graph in enumerate(self.graph_data.graphs):
            data = torch_geometric.data.Data()
            data_x = torch.zeros((graph.number_of_nodes(), num_node_labels))
            # create one hot encoding for node labels
            for j, node in graph.nodes(data=True):
                data_x[j][node['label']] = 1
            data.x = data_x
            edge_index = torch.zeros((2, 2 * len(graph.edges)), dtype=torch.long)
            # add each edge twice, once in each direction
            for j, edge in enumerate(graph.edges):
                edge_index[0][2 * j] = edge[0]
                edge_index[1][2 * j] = edge[1]
                edge_index[0][2 * j + 1] = edge[1]
                edge_index[1][2 * j + 1] = edge[0]

            data.edge_index = edge_index
            data.y = torch.tensor(self.graph_data.graph_labels[i])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def zinc_to_graph_data(train, validation, test, graph_db_name, use_features=True):
    graphs = GraphData()
    graphs.graph_db_name = graph_db_name
    graphs.edge_labels['primary'] = EdgeLabels()
    graphs.node_labels['primary'] = NodeLabels()
    graphs.node_labels['primary'].node_labels = []
    graphs.edge_labels['primary'].edge_labels = []
    graphs.graph_labels = []
    graphs.one_hot_labels = []
    graphs.max_nodes = 0
    graphs.num_classes = 1
    graphs.num_graphs = len(train) + len(validation) + len(test)

    max_label = 0
    label_set = set()

    original_source = -1
    for data in [train, validation, test]:
        for i, graph in enumerate(data):
            # add new graph
            graphs.graphs.append(nx.Graph())
            graphs.edge_labels['primary'].edge_labels.append([])
            graphs.inputs.append(torch.ones(graph['x'].shape[0]).float())
            # add graph inputs using the values from graph['x'] and flatten the tensor
            if use_features:
                graphs.inputs[-1] = graph['x'].flatten().float()

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

            # update max_label
            max_label = max(abs(max_label), max(abs(graph['x'])).item())
            # add graph label
            for node_label in graph['x']:
                label_set.add(node_label.item())

            graphs.edge_labels['primary'].edge_labels.append(graph['edge_attr'])
            graphs.graph_labels.append(graph['y'].item())
            graphs.one_hot_labels.append(graph['y'].float())
            graphs.max_nodes = max(graphs.max_nodes, len(graph['x']))

            pass
        pass
    if use_features:
        # normalize graph inputs
        number_of_node_labels = len(label_set)
        label_set = sorted(label_set)
        step = 1.0 / number_of_node_labels
        for i, graph in enumerate(graphs.inputs):
            for j, val in enumerate(graph):
                graphs.inputs[i][j] = (label_set.index(val) + 1) * step * (-1) ** label_set.index(val)

    # convert one hot label list to tensor
    graphs.one_hot_labels = torch.stack(graphs.one_hot_labels)
    return graphs
