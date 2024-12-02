import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Callable, List

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from numpy.ma.core import shape
from torch_geometric.data import InMemoryDataset, Data, TensorAttr
from torch_geometric.data.data import BaseData
from torch_geometric.datasets import ZINC, TUDataset, GNNBenchmarkDataset

from src.utils import NodeLabeling, EdgeLabeling
from src.utils.GraphLabels import NodeLabels, EdgeLabels, Properties
from src.utils.utils import load_graphs
from torch_geometric.io import fs, read_tu_data
from torch_geometric.utils.convert import to_networkx




class RuleGNNDataset(InMemoryDataset):
    def __init__(
            self,
            root: str,
            name: str,
            transform: Optional[Callable] = None,
            pre_transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            from_existing_data: Optional[str] = None,
            force_reload: bool = False,
            use_node_attr: bool = True,
            use_edge_attr: bool = True,
            delete_zero_columns: bool = True,
            precision: str = 'float',
            input_features = None,
            output_features = None,
            task = None
    ) -> None:
        self.name = name
        self.from_existing_data = from_existing_data
        self.nx_graphs = []
        self.unique_node_labels = 0
        self.node_labels = {}
        self.edge_labels = {}
        self.properties = {}
        self.precision = torch.float
        if precision == 'double':
            self.precision = torch.double
        super(RuleGNNDataset, self).__init__(root, transform, pre_transform, force_reload=force_reload)
        out = fs.torch_load(self.processed_paths[0])
        if not isinstance(out, tuple) or len(out) < 3:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        assert len(out) == 3 or len(out) == 4

        if len(out) == 3:  # Backward compatibility.
            data, self.slices, self.sizes = out
            data_cls = Data
        else:
            data, self.slices, self.sizes, data_cls = out

        self._num_graph_nodes = torch.zeros(len(self), dtype=torch.long)
        num_node_attributes = self.num_node_attributes
        num_edge_attributes = self.num_edge_attributes

        # split node labels and attributes as well as edge labels and attributes
        if data.get('x', None) is not None:
            if delete_zero_columns:
                # remove columns with only zeros
                if self.precision == torch.float:
                    data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].float()
                else:
                    data['x'] = data['x'][:, data['x'].sum(dim=0) != 0].double()
                self.sizes['num_node_labels'] = data['x'].shape[1]
        else:
            if data.get('num_nodes', None) is None:
                data['num_nodes'] = torch.zeros(len(self), dtype=torch.long)
            # create data['x'] using vectors of ones
            data['x'] = torch.ones(data['num_nodes'], 1, dtype=self.precision)
            self.sizes['num_node_labels'] = 1
            self.slices['x'] = [0]
            self.slices['x'] += data['_num_nodes']
            self.slices['x'] = torch.tensor(self.slices['x'], dtype=torch.long).cumsum(dim=0)

        if data.get('x', None) is not None:
            if data['x'].shape[1] == 1:
                self.node_labels['primary'] = data['x'].clone().detach().long()
            else:
                self.node_labels['primary'] = torch.argmax(data['x'][:, num_node_attributes:], dim=1)
            self.unique_node_labels = torch.unique(self.node_labels['primary']).shape[0]
            if not use_node_attr:
                num_node_attributes = self.num_node_attributes
                data['x'] = data['x'][:, num_node_attributes:]



        if data.get('edge_attr', None) is not None:
            self.edge_labels['primary'] = torch.argmax(data['edge_attr'][:, num_edge_attributes:], dim=1)
            if not use_edge_attr:
                num_edge_attrs = self.num_edge_attributes
                data['edge_attr'] = data['edge_attr'][:, num_edge_attrs:]

        data['num_nodes'] = torch.zeros(len(self), dtype=torch.long)
        for i in range(len(self)):
            data['num_nodes'][i] = data['x'][self.slices['x'][i]:self.slices['x'][i+1]].shape[0]


        self.preprocess_rule_gnn_data(data, input_features, output_features, task=task)


        if not isinstance(data, dict):  # Backward compatibility.
            self.data = data
        else:
            # split node labels and attributes as well as edge labels and attributes
            self.data = data_cls.from_dict(data)


        assert isinstance(self._data, Data)

    @property
    def raw_dir(self) -> str:
        name = f'raw'
        return os.path.join(self.root, self.name, name)

    @property
    def processed_dir(self) -> str:
        name = f'processed'
        return os.path.join(self.root, self.name, name)

    @property
    def num_node_labels(self) -> int:
        return self.sizes['num_node_labels']

    @property
    def num_node_attributes(self) -> int:
        return self.sizes['num_node_attributes']

    @property
    def num_edge_labels(self) -> int:
        return self.sizes['num_edge_labels']

    @property
    def num_edge_attributes(self) -> int:
        return self.sizes['num_edge_attributes']

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator']
        return [f'{self.name}_{name}.txt' for name in names]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        sizes = None
        if self.from_existing_data is not None:
            if self.from_existing_data == 'ZINC':
                data = ZINC(root='tmp/', subset=True, split='train')
                self.data, self.slices, sizes = data._data, data.slices, data.sizes
            elif self.from_existing_data == 'TUDataset':
                tu_dataset = TUDataset(root='tmp/', name=self.name, use_node_attr=True, use_edge_attr=True)
                self.data, self.slices, sizes = tu_dataset._data, tu_dataset.slices, tu_dataset.sizes
            elif self.from_existing_data == 'NEL':
                self.data, self.slices, sizes = self.read_nel_data_v2()
            elif self.from_existing_data == 'gnn_benchmark':
                dataset = GNNBenchmarkDataset("tmp/", self.name)
                sizes = {
                    'num_node_labels': dataset.num_features,
                    'num_node_attributes': dataset.num_node_features,
                    'num_edge_labels': dataset.num_edge_features,
                    'num_edge_attributes': 0
                }
                # add x to data uing num_nodes times 0
                num_graphs = len(dataset.data.y)
                dataset.data.x = torch.ones(dataset.data.num_nodes, 1)
                nodes_per_graph = dataset.data.num_nodes // num_graphs
                # remove num_nodes from x
                dataset.slices['x'] = torch.linspace(0, dataset.data.num_nodes, num_graphs + 1, dtype=torch.long)
                self.slices = dataset.slices
                self.data = dataset.data
                pass
        else:
            print('Cannot process the data')

        if self.pre_filter is not None or self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            self.data, self.slices = self.collate(data_list)
            self._data_list = None  # Reset cache.

        assert isinstance(self._data, Data)
        fs.torch_save(
            (self._data.to_dict(), self.slices, sizes, self._data.__class__),
            self.processed_paths[0],
        )

    def read_nel_data(self):
        graphs, labels = load_graphs(Path(self.raw_dir), self.name, graph_format='NEL')
        node_labels = []
        node_attributes = []
        node_slices = [0]
        with_node_attributes = False
        for graph_id, graph in enumerate(graphs):
            print(f'Processing graph {graph_id+1}/{len(graphs)}')
            node_labels += [0] * graph.number_of_nodes()
            node_attributes += [0] * graph.number_of_nodes()
            for node in graph.nodes(data=True):
                if 'label' in node[1]:
                    index_start = np.sum(node_slices[0:graph_id+1])
                    node_labels[index_start+node[0]] = int(node[1]['label'][0])
                    if len(node[1]['label']) > 1:
                        with_node_attributes = True
                        node_attributes[index_start+node[0]] = node[1]['label'][1:]
            node_slices.append(graph.number_of_nodes())
        # convert the node labels to a tensor
        node_labels = torch.tensor(node_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        node_labels = torch.nn.functional.one_hot(node_labels).float()
        # convert the node attributes to a tensor
        node_attributes = torch.tensor(node_attributes, dtype=torch.float)
        if len(node_attributes) == 0 or not with_node_attributes:
            node_attributes = None
        if node_attributes is not None:
            # stack node attributes and node labels together to form the node feature matrix
            x = torch.cat((node_attributes, node_labels), dim=1)
        else:
            x = node_labels
        node_slices = torch.tensor(node_slices, dtype=torch.long).cumsum(dim=0)
        # create edge_index tensor
        edge_indices = []
        edge_slices = [0]
        edge_labels = []
        edge_attributes = []
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                edge_indices.append([edge[0], edge[1]])
                if 'label' in edge[2]:
                    edge_labels.append(int(edge[2]['label'][0]))
                    if len(edge[2]['label']) > 1:
                        edge_attributes.append(edge[2]['label'][1:])
            edge_slices.append(len(graph.edges()))
        # convert the edge indices to a tensor
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).T
        edge_slices = torch.tensor(edge_slices, dtype=torch.long).cumsum(dim=0)
        # convert the edge labels to a tensor
        edge_labels = torch.tensor(edge_labels, dtype=torch.long)
        # apply row-wise one-hot encoding
        edge_labels = torch.nn.functional.one_hot(edge_labels).float()
        # convert the edge attributes to a tensor
        edge_attributes = torch.tensor(edge_attributes, dtype=torch.float)
        if len(edge_attributes) == 0:
            edge_attributes = None
        if edge_attributes is not None:
            # stack edge attributes and edge labels together to form the edge feature matrix
            edge_attr = torch.cat((edge_attributes, edge_labels), dim=1)
        else:
            edge_attr = edge_labels
        y = torch.tensor(labels, dtype=torch.long)
        y_slices = torch.arange(0, len(labels)+1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attr, y=y)
        slices = {'edge_index': edge_slices,
                    'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}
        sizes = {'num_node_labels': node_labels.shape[1],
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attributes.shape[1] if edge_attributes is not None else 0}
        return data, slices, sizes

    def read_nel_data_v2(self):
        load_path = Path(self.raw_dir)
        # load the nodes from the file
        node_labels = []
        node_attributes = None
        node_slices = [0]
        node_counter = 0
        unique_labels = []
        with open(load_path.joinpath(self.name + "_Nodes.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            # convert into torch tensor
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing node {i+1}/{len(lines)} in dataset {self.name}')
                data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, data)))
            graph_ids = torch_lines[:, 0].long()
            # get slice vector from unique graph ids
            node_slices = torch.unique(graph_ids, return_counts=True)[1]
            # add 0 at the beginning
            node_slices = torch.cat((torch.tensor([0]), node_slices)).cumsum(dim=0).long()
            node_ids = torch_lines[:, 1].long()
            node_labels = torch_lines[:, 2].long()
            unique_labels = len(torch.unique(node_labels))
            node_attr = None
            # sort the node labels graph-wise (node_slices) according to the node ids
            for idx in range(len(node_slices) - 1):
                sorted_indices = torch.argsort(node_ids[node_slices[idx]:node_slices[idx+1]])
                node_labels[node_slices[idx]:node_slices[idx+1]] = node_labels[node_slices[idx]:node_slices[idx+1]][sorted_indices]
                if line_length > 3:
                    node_attr = torch_lines[:, 3:]
                    node_attr[node_slices[idx]:node_slices[idx+1]] = node_attr[node_slices[idx]:node_slices[idx+1]][sorted_indices]

        x = None
        # one hot encoding if number of node labels is smaller than 100
        if unique_labels < 100:
            x = torch.nn.functional.one_hot(node_labels).float()
        else:
            x = node_labels
        if node_attr is not None:
            x = torch.cat((node_attr, x), dim=1)

        edge_indices = None
        edge_slices = None
        edge_labels = None
        edge_attr = None
        with open(load_path.joinpath(self.name + "_Edges.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length), dtype=torch.float)
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    print(f'Processing edge {i+1}/{len(lines)} in dataset {self.name}')
                data = line.strip().split(" ")
                torch_lines[i] = torch.tensor(list(map(float, data)))
            graph_ids = torch_lines[:, 0].long()
            # get slice vector from unique graph ids
            edge_slices = torch.unique(graph_ids, return_counts=True)[1]
            # add 0 at the beginning
            edge_slices = torch.cat((torch.tensor([0]), edge_slices)).cumsum(dim=0).long()
            edge_indices = torch_lines[:, 1:3].long().T
            edge_labels = torch_lines[:, 3].long()
            edge_labels = torch.nn.functional.one_hot(edge_labels).float()
            edge_attr = None
            if line_length > 4:
                edge_attr = torch_lines[:, 4:]

        y = None
        with open(load_path.joinpath(self.name + "_Labels.txt"), "r") as f:
            lines = f.readlines()
            line_length = len(lines[0].strip().split(" "))
            torch_lines = torch.zeros((len(lines), line_length - 1), dtype=torch.long)
            for i, line in enumerate(lines):
                data = line.strip().split(" ")
                graph_name = data[0]
                torch_lines[i] = torch.tensor(list(map(float, data[1:])))
            y = torch_lines[:, 0].long()


        y_slices = torch.arange(0, len(y) + 1, dtype=torch.long)
        data = Data(x=x, edge_index=edge_indices, edge_attr=edge_attr, y=y)
        slices = {'edge_index': edge_slices,
                  'x': node_slices,
                  'edge_attr': edge_slices.detach().clone(),
                  'y': y_slices}

        sizes = {'num_node_labels': unique_labels,
                 'num_node_attributes': node_attributes.shape[1] if node_attributes is not None else 0,
                 'num_edge_labels': edge_labels.shape[1],
                 'num_edge_attributes': edge_attr.shape[1] if edge_attr is not None else 0}
        return data, slices, sizes

    def create_nx_graphs(self, directed: bool = False):
        self.nx_graphs = []
        counter = 0
        for g_id, graph in enumerate(self):
            self.nx_graphs.append(to_networkx(
                data=graph,
                node_attrs=['x'],
                edge_attrs=['edge_attr'] if graph.edge_attr is not None else None,
                to_undirected=not directed))
            # change node label 'x' to 'primary_label'
            for node in self.nx_graphs[-1].nodes(data=True):
                self.nx_graphs[-1].nodes[node[0]]['primary_label'] = self.node_labels['primary'][counter].item()
                del self.nx_graphs[-1].nodes[node[0]]['x']
                counter += 1
        pass

    def preprocess_rule_gnn_data(self, data, input_features=None, output_features=None, task=None) -> None:
        if input_features is not None and task is not None:
            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)
            use_features_as_channels = input_features.get('features_as_channels', False)

            ### Determine the input data
            if use_labels:
                data['x'] = data['x'][:, self.num_node_attributes:]
                if transformation in ['one_hot', 'one_hot_encoding']:
                    pass
                else:
                    data['x'] = torch.argmax(data['x'], dim=1).unsqueeze(1)
            elif use_constant:
                data['x'] = torch.full(size=(data['x'].shape[0], input_features.get('in_dimensions', 1)), fill_value=input_features.get('value', 1.0), dtype=self.precision)
            elif use_features:
                data['x'] = data['x'][:, :self.num_node_attributes]
            elif use_labels_and_features:
                # get first self.num_node_attributes columns and on the rest apply argmax
                data['x'] = torch.cat((data['x'][:, :self.num_node_attributes], torch.argmax(data['x'][:,self.num_node_attributes:], dim=1).unsqueeze(0)), dim=1)
            else:
                pass

            # normalize the graph input labels, i.e. to have values between -1 and 1, no zero values
            if use_labels and transformation == 'normalize':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values
                data['x'] = data['x'].apply_(lambda x: normalized_node_labels[x])
            elif use_labels and transformation == 'unit_circle':
                '''
                Arrange the labels in an 2D unit circle
                '''
                num_node_labels = self.unique_node_labels
                # duplicate data column
                data['x'] = data['x'].repeat(1, 2)
                data['x'] = data['x'][:, 0:1].apply_(lambda x: torch.cos(2 * np.pi * x / num_node_labels))
                data['x'] = data['x'][:, 1:2].apply_(lambda x: torch.sin(2 * np.pi * x / num_node_labels))
            elif use_labels_and_features and transformation == 'normalize_labels':
                # get the number of unique node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                if num_node_labels % 2 == 1:
                    num_node_labels += 1
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # if value is even, add 1 to make it odd
                    if value % 2 == 0:
                        value = ((value + 1) * interval_length)
                    else:
                        value = (-1) * (value * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])
            elif use_labels_and_features and transformation == 'normalize_positive':
                # get the number of different node labels
                num_node_labels = self.unique_node_labels
                # get the next even number if the number of node labels is odd
                intervals = num_node_labels + 1
                interval_length = 1.0 / intervals
                normalized_node_labels = torch.zeros(self.num_node_labels)
                for idx, entry in enumerate(normalized_node_labels):
                    value = idx
                    value = int(value)
                    # map the value to the interval [0,1]
                    value = ((value + 1) * interval_length)
                    normalized_node_labels[idx] = value
                # replace values in data['x'] by the normalized values only for the last column
                data['x'] = data['x'][:, -1].apply_(lambda x: normalized_node_labels[x])


            # Determine the output data
            #if task == 'regression':
            #    self.num_classes = 1
            #    if type(self.graph_labels[0]) == list:
            #        self.num_classes = len(self.graph_labels[0])
            #else:
            #    try:
            #        self.num_classes = len(set(self.graph_labels))
            #    except:
            #        self.num_classes = len(self.graph_labels[0])
            #
            # one hot encode y

            if task == 'regression':
                if output_features.get('transformation', None) is not None:
                    data['y'] = transform_data(data['y'], output_features)
            return None

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'

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


def transform_data(data, transformation_dict: Dict[str, Dict[str, str]]):
    # reformat the output data, shift to positive values and make the values smaller
    transformation_args = [{}]
    if transformation_dict.get('transformation_args', None) is not None:
        transformation_args = transformation_dict['transformation_args']
    if type(transformation_dict['transformation']) == list:
        for i, expression in enumerate(transformation_dict['transformation']):
            data = eval(expression)(input=data, **transformation_args[i])
    else:
        data = eval(transformation_dict['transformation'])(input=data, **transformation_args)
    return data


class GraphData:
    def __init__(self):
        self.name = ''
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

    def __len__(self):
        return len(self.graphs)

    def __iadd__(self, other):
        '''
        Add another GraphData object to this one.
        '''
        if 'Union' in self.name:
            pass
        else:
            self.name = f'Union_{self.name}'
        self.name += f'_{other.name}'
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

    def load_nel_graphs(self, db_name: str, path: Path, input_features=None, output_features=None, task=None, only_graphs=False):
        self.name = db_name
        self.graphs, self.graph_labels = load_graphs(path.joinpath(Path(f'{db_name}/raw/')), db_name, graph_format='NEL')
        self.num_graphs = len(self.graphs)
        self.avg_nodes = sum([g.number_of_nodes() for g in self.graphs]) / self.num_graphs
        self.avg_degree = sum([g.number_of_edges() for g in self.graphs]) / self.num_graphs

        self.max_nodes = max([g.number_of_nodes() for g in self.graphs])

        self.add_node_labels(node_labeling_name='primary', node_labeling_method=NodeLabeling.standard_node_labeling)
        self.add_edge_labels(edge_labeling_name='primary', edge_labeling_method=EdgeLabeling.standard_edge_labeling)

        if not only_graphs:
            if input_features is None:
                input_features = {'name': 'node_labels', 'transformation': {'name': 'normalize'}}
            if output_features is None:
                output_features = {}

            use_labels = input_features.get('name', 'node_labels') == 'node_labels'
            use_constant = input_features.get('name', 'node_labels') == 'constant'
            use_features = input_features.get('name', 'node_labels') == 'node_features'
            use_labels_and_features = input_features.get('name', 'node_labels') == 'all'
            transformation = input_features.get('transformation', None)

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
                self.output_data = self.output_data.unsqueeze(1)
                if output_features.get('transformation', None) is not None:
                    self.output_data = transform_data(self.output_data, output_features)

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
        """
        Adapt the precision of the input data
        :param precision: str - precision of the input data (double or float)
        """
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
            self.graph_name_to_index[graph.name] = indices






        self.graph_data = graph_data


def get_graph_data(db_name: str, data_path : Path, task='graph_classification', input_features=None, output_features=None, graph_format='NEL', only_graphs=False, precision='double'):
    """
    Load the graph data by name.
    :param db_name: str - name of the graph database
    :param data_path: Path - path to the data
    :param task: str - task to perform on the data
    :param input_features: dict - input features
    :param output_features: dict - output features
    :param graph_format: str - format of the data NEL: node edge label format
    :param only_graphs: bool - whether to load only the graphs

    """
    # load the graph data
    if graph_format == 'NEL':
        graph_data = GraphData()
        graph_data.load_nel_graphs(db_name=db_name, path=data_path, input_features=input_features, output_features=output_features, task=task, only_graphs=only_graphs)
    elif graph_format == 'RuleGNNDataset':
        graph_data = RuleGNNDataset(root=str(data_path), name=db_name, precision=precision, input_features=input_features, output_features=output_features, task=task)
        pass
    else:
        raise ValueError(f'Graph format {graph_format} not supported')
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
    graphs.name = graph_db_name
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
