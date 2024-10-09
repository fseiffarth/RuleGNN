import os
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from numpy.ma.core import shape
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import ZINC

from src.TrainTestData import TrainTestData as ttd
import src.utils.ReadWriteGraphs.GraphDataToGraphList as gdtgl
from src.utils import NodeLabeling, EdgeLabeling
from src.utils.GraphLabels import NodeLabels, EdgeLabels, Properties
from src.utils.utils import load_graphs


def relabel_most_frequent(labels: NodeLabels, num_max_labels: int):
    if num_max_labels is None:
        num_max_labels = -1
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

class GraphData:
    def __init__(self):
        self.graph_db_name = ''
        self.graphs = []
        self.input_data = []
        self.node_labels: Dict[str, NodeLabels] = {}
        self.edge_labels: Dict[str, EdgeLabels] = {}
        self.properties: Dict[str, Properties] = {}
        self.graph_labels = []
        self.output_data = []
        self.num_classes = 0
        self.max_nodes = 0
        self.num_graphs = 0
        self.input_feature_dimensions = 1
        self.input_channels = 1
        self.output_feature_dimensions = 1
        self.avg_nodes = 0
        self.avg_degree = 0

    def __iadd__(self, other):
        '''
        Add another GraphData object to this one.
        '''
        if 'Union' in self.graph_db_name:
            pass
        else:
            self.graph_db_name = f'Union_{self.graph_db_name}'
        self.graph_db_name += f'_{other.graph_db_name}'
        self.graphs += other.graphs
        self.input_data += other.input_data

        for key, value in other.node_labels.items():
            if key in self.node_labels:
                self.node_labels[key] += value
            else:
                self.node_labels[key] = value

        for key, value in other.edge_labels.items():
            if key in self.edge_labels:
                self.edge_labels[key] += value
            else:
                self.edge_labels[key] = value

        for key, value in other.properties.items():
            if key in self.properties:
                self.properties[key] += value
            else:
                self.properties[key] = value


        self.graph_labels += other.graph_labels
        self.output_data += other.output_data
        self.num_classes = max(self.num_classes, other.num_classes)
        self.max_nodes = max(self.max_nodes, other.max_nodes)

    def add_node_labels(self, node_labeling_name, max_labels=-1, node_labeling_method=None, **kwargs) -> None:
        if node_labeling_method is not None:
            node_labeling = NodeLabels()
            node_labeling.node_labels, node_labeling.unique_node_labels, node_labeling.db_unique_node_labels = node_labeling_method(
                self.graphs, **kwargs)
            node_labeling.num_unique_node_labels = max(1, len(node_labeling.db_unique_node_labels))

            key = node_labeling_name
            if max_labels is not None and max_labels > 0:
                key = f'{node_labeling_name}_{max_labels}'

            self.node_labels[key] = node_labeling
            relabel_most_frequent(self.node_labels[key], max_labels)

    def add_edge_labels(self, edge_labeling_name, edge_labeling_method=None, **kwargs) -> None:
        if edge_labeling_method is not None:
            edge_labeling = EdgeLabels()
            edge_labeling.edge_labels, edge_labeling.unique_edge_labels, edge_labeling.db_unique_edge_labels = edge_labeling_method(
                self.graphs, **kwargs)
            edge_labeling.num_unique_edge_labels = max(1, len(edge_labeling.db_unique_edge_labels))
            self.edge_labels[edge_labeling_name] = edge_labeling

    def load_nel_graphs(self, db_name: str, path: Path, input_features=None, task=None, only_graphs=False):
        self.graph_db_name = db_name
        self.graphs, self.graph_labels = load_graphs(path.joinpath(Path(f'{db_name}/raw/')), db_name, graph_format='NEL')
        self.num_graphs = len(self.graphs)
        self.avg_nodes = sum([g.number_of_nodes() for g in self.graphs]) / self.num_graphs
        self.avg_degree = sum([g.number_of_edges() for g in self.graphs]) / self.num_graphs

        self.max_nodes = 0
        for g in self.graphs:
            self.max_nodes = max(self.max_nodes, g.number_of_nodes())

        self.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        self.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)

        if not only_graphs:
            if input_features is None:
                input_features = {'name': 'node_labels', 'transformation': {'name': 'normalize'}}

            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)
            use_features_as_channels = input_features.get('features_as_channels', False)

            ### Determine the input data
            self.input_data = []
            ## add node labels
            for graph_id, graph in enumerate(self.graphs):
                if use_labels:
                    if transformation in ['one_hot', 'one_hot_encoding']:
                        self.input_data.append(torch.zeros(1,graph.number_of_nodes(), self.node_labels['primary'].num_unique_node_labels))
                        for node in graph.nodes(data=True):
                            self.input_data[-1][0][node[0]][self.node_labels['primary'].node_labels[graph_id][node[0]]] = 1
                    else:
                        self.input_data.append(torch.ones(1,graph.number_of_nodes(),1).float())
                        for node in graph.nodes(data=True):
                            self.input_data[-1][0][node[0]] = self.node_labels['primary'].node_labels[graph_id][node[0]]
                elif use_constant:
                    self.input_data.append(torch.full(size=(1,graph.number_of_nodes(),1), fill_value=input_features.get('value', 1.0)).float())
                elif use_features:
                    self.input_data.append(torch.zeros(1,graph.number_of_nodes(), len(graph.nodes(data=True)[0]['label'][1:])))
                    for node in graph.nodes(data=True):
                        # add all except the first element of the label
                        self.input_data[-1][0][node[0]] = torch.tensor(node[1]['label'][1:])
                elif use_labels_and_features:
                    self.input_data.append(torch.zeros(1,graph.number_of_nodes(), len(graph.nodes(data=True)[0]['label'])))
                    for node in graph.nodes(data=True):
                        # add all except the first element of the label
                        self.input_data[-1][0][node[0]] = torch.tensor([self.node_labels['primary'].node_labels[graph_id][node[0]]] + node[1]['label'][1:])



            # normalize the graph input labels, i.e. to have values between -1 and 1, no zero values
            if use_labels and transformation == 'normalize':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][0][j]
                        # get integer value of the node label
                        value = int(value)
                        # if value is even, add 1 to make it odd
                        if value % 2 == 0:
                            value = ((value + 1) * interval_length)
                        else:
                            value = (-1) * (value * interval_length)
                        self.input_data[i][0][j] = value
            elif use_labels and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][0][j]
                        # get integer value of the node label
                        value = int(value)
                        # map the value to the interval [0,1]
                        value = ((value + 1) * interval_length)
                        self.input_data[i][0][j] = value


            elif use_labels and transformation == 'unit_circle':
                '''
                Arange the labels in an 2D unit circle
                # TODO: implement this
                '''
                updated_input_data = []
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                for i, graph in enumerate(self.graphs):
                    updated_input_data.append(torch.ones(1, graph.number_of_nodes(), 2))
                    for j in range(graph.number_of_nodes()):
                        value = int(self.input_data[i][0][j])
                        # get integer value of the node label
                        value = int(value)
                        updated_input_data[-1][0][j][0] = np.cos(2*np.pi*value / num_node_labels)
                        updated_input_data[-1][0][j][1] = np.sin(2*np.pi*value / num_node_labels)
                self.input_data = updated_input_data
            elif use_labels_and_features and transformation == 'normalize_labels':
                # get the number of different node labels
                num_node_labels = self.node_labels['primary'].num_unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                for i, graph in enumerate(self.graphs):
                    for j in range(graph.number_of_nodes()):
                        value = self.input_data[i][j][0]
                        # get integer value of the node label
                        value = int(value)
                        # if value is even, add 1 to make it odd
                        if value % 2 == 0:
                            value = ((value + 1) * interval_length)
                        else:
                            value = (-1) * (value * interval_length)
                        self.input_data[i][j][0] = value

            if use_features_as_channels:
                # swap the dimensions
                for i in range(len(self.input_data)):
                    self.input_data[i] = self.input_data[i].permute(2,1,0)


            # Determine the output data
            if task == 'regression':
                self.num_classes = 1
                if type(self.graph_labels[0]) == list:
                    self.num_classes = len(self.graph_labels[0])
            else:
                try:
                    self.num_classes = len(set(self.graph_labels))
                except:
                    self.num_classes = len(self.graph_labels[0])

            self.output_data = torch.zeros(self.num_graphs, self.num_classes)

            if task == 'regression':
                self.output_data = torch.tensor(self.graph_labels)
                # reformat the output data, add minimum +0.01 to avoid zero values and take the log
                self.output_data = torch.pow(torch.exp(self.output_data), 1/4)
                self.output_feature_dimensions = 1
            else:
                for i, label in enumerate(self.graph_labels):
                    if type(label) == int:
                        self.output_data[i][label] = 1
                    elif type(label) == list:
                        self.output_data[i] = torch.tensor(label)
                # the output feature dimension
                self.output_feature_dimensions = self.output_data.shape[1]
            # the input channel dimension
            self.input_channels = self.input_data[0].shape[0]
            # the input feature dimension
            self.input_feature_dimensions = self.input_data[0].shape[2]
        return None

    def set_precision(self, precision:str='double'):
        # adapt the precision of the input data
        if precision == 'double':
            for i in range(len(self.input_data)):
                self.input_data[i] = self.input_data[i].double()
            self.output_data = self.output_data.double()
        elif precision == 'float':
            for i in range(len(self.input_data)):
                self.input_data[i] = self.input_data[i].float()
            self.output_data = self.output_data.float()


class GraphDataUnion:
    def __init__(self, db_names, graph_data):
        self.graph_db_names = db_names
        self.graph_name_to_index = {}

        # merge all the graph data into one
        self.graph_data = GraphData()
        start_index = 0
        for i, graph in enumerate(graph_data):
            if i == 0:
                self.graph_data = graph
            else:
                self.graph_data += graph
            indices = np.arange(start_index, start_index + len(graph))
            start_index += len(graph)
            self.graph_name_to_index[graph.graph_db_name] = indices






        self.graph_data = graph_data


def get_graph_data(db_name: str, data_path : Path, task='graph_classification', input_features=None, graph_format='NEL', only_graphs=False):
    """
    Load the graph data by name.
    :param db_name: str - name of the graph database
    :param data_path: Path - path to the data
    :param task: str - task to perform on the data
    :param input_features: dict - input features
    :param relabel_nodes: bool - whether to relabel nodes
    :param graph_format: str - format of the data NEL: node edge label format
    :param only_graphs: bool - whether to load only the graphs

    """
    # load the graph data
    graph_data = GraphData()
    if graph_format == 'NEL':
        graph_data.load_nel_graphs(db_name=db_name, path=data_path, input_features=input_features, task=task, only_graphs=only_graphs)
    # else:
    #     if db_name == 'CSL_original':
    #         from src.Preprocessing.csl import CSL
    #         csl = CSL()
    #         graph_data = csl.get_graphs(with_distances=False)
    #     elif db_name == 'CSL':
    #         graph_data = GraphData()
    #         graph_data.load_nel_graphs(db_name, data_path, use_labels)
    #     elif db_name == 'ZINC_original':
    #         zinc_train = ZINC(root="../../ZINC_original/", subset=True, split='train')
    #         zinc_val = ZINC(root="../../ZINC_original/", subset=True, split='val')
    #         zinc_test = ZINC(root="../../ZINC_original/", subset=True, split='test')
    #         graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC_original")
    #     elif db_name == 'ZINC':
    #         graph_data = GraphData()
    #         graph_data.load_nel_graphs(db_name, data_path, use_labels, task='regression')
    #     elif ('LongRings' in db_name) or ('EvenOddRings' in db_name) or ('SnowflakesCount' in db_name) or (
    #             'Snowflakes' in db_name):
    #         graph_data = GraphData()
    #         # add db_name and raw to the data path
    #         data_path = data_path.joinpath(db_name + "/raw/")
    #         graph_data.load_nel_graphs(db_name, data_path, use_labels)
    #     else:
    #         graph_data = GraphData()
    #         graph_data.init_from_graph_db(data_path, db_name, relabel_nodes=relabel_nodes, use_features=use_labels,
    #                                       use_attributes=use_attributes)
    return graph_data


class BenchmarkDatasets(InMemoryDataset):
    def __init__(self, root: str, name: str, graph_data: GraphData):
        self.graph_data = graph_data
        self.name = name
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=True)

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
    graphs.output_data = []
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
            # add new nodes

            #graphs.edge_labels['primary'].edge_labels.append([])
            graphs.input_data.append(torch.ones(graph['x'].shape[0]).float())
            # add graph inputs using the values from graph['x'] and flatten the tensor
            if use_features:
                graphs.input_data[-1] = graph['x'].flatten().float()

            edges = graph['edge_index']
            # format edges to list of tuples
            edges = edges.T.tolist()
            # add edges to graph
            for i, edge in enumerate(edges):
                if edge[0] < edge[1]:
                    edge_label = graph['edge_attr'][i].item()
                    graphs.graphs[-1].add_edge(edge[0], edge[1], label=edge_label)
                    #graphs.edge_labels['primary'].edge_labels[-1].append(edge_label)
            # add node labels
            graphs.node_labels['primary'].node_labels.append([x.item() for x in graph['x']])
            # add also node labels to the existing graph node
            for node in graphs.graphs[-1].nodes(data=True):
                node[1]['label'] = graph['x'][node[0]].item()

            # update max_label
            max_label = max(abs(max_label), max(abs(graph['x'])).item())
            # add graph label
            for node_label in graph['x']:
                label_set.add(node_label.item())

            graphs.edge_labels['primary'].edge_labels.append(graph['edge_attr'])
            graphs.graph_labels.append(graph['y'].item())
            graphs.output_data.append(graph['y'].float())
            graphs.max_nodes = max(graphs.max_nodes, len(graph['x']))

            pass
        pass
    if use_features:
        # normalize graph inputs
        number_of_node_labels = len(label_set)
        label_set = sorted(label_set)
        step = 1.0 / number_of_node_labels
        for i, graph in enumerate(graphs.input_data):
            for j, val in enumerate(graph):
                graphs.input_data[i][j] = (label_set.index(val) + 1) * step * (-1) ** label_set.index(val)

    # convert one hot label list to tensor
    graphs.output_data = torch.stack(graphs.output_data)
    return graphs
