import ast
import gzip
import os
import pickle
from collections import OrderedDict
from typing import List, Tuple

import yaml

from src.utils.utils import convert_to_tuple


class NodeLabels:
    def __init__(self, node_labels=None):
        self.node_labels = None
        self.unique_node_labels = None
        self.db_unique_node_labels = None
        self.num_unique_node_labels = 0

        if node_labels is not None:
            self.node_labels = node_labels
            self.db_unique_node_labels = {}
            self.unique_node_labels = []
            for g_labels in node_labels:
                self.unique_node_labels.append({})
                for l in g_labels:
                    if l not in self.db_unique_node_labels:
                        self.db_unique_node_labels[l] = 1
                    else:
                        self.db_unique_node_labels[l] += 1
                    if l not in self.unique_node_labels[-1]:
                        self.unique_node_labels[-1][l] = 1
                    else:
                        self.unique_node_labels[-1][l] += 1
            self.num_unique_node_labels = len(self.db_unique_node_labels)

    def __iadd__(self, other):
        pass



def combine_node_labels(labels: List[NodeLabels]):
    # create tuples for each node
    node_labels = []
    label_map = {}
    for i, g_labels in enumerate(labels[0].node_labels):
        node_labels.append([])
        for j, l in enumerate(g_labels):
            label_tuple = []
            for k in range(len(labels)):
                label_tuple.append(labels[k].node_labels[i][j])
            if tuple(label_tuple) not in label_map:
                label_map[tuple(label_tuple)] = 1
            else:
                label_map[tuple(label_tuple)] += 1
            node_labels[-1].append(tuple(label_tuple))

    # sort dict by values
    label_map = OrderedDict(sorted(label_map.items(), key=lambda item: item[1], reverse=True))

    index_map = {}
    # iterate over ordered dict and create map
    for i, key in enumerate(label_map):
        index_map[key] = i

    # create new labels from node labels using the map
    new_labels = []
    for g_labels in node_labels:
        new_labels.append([])
        for l in g_labels:
            new_labels[-1].append(index_map[l])

    return NodeLabels(new_labels)


class EdgeLabels:
    def __init__(self):
        self.edge_labels = None
        self.unique_edge_labels = None
        self.db_unique_edge_labels = None
        self.num_unique_edge_labels = 0


class Properties:
    def __init__(self, path: str, db_name: str, property_name: str, valid_values: dict[tuple[int, int], list[int]]):
        self.name = property_name
        self.db = db_name
        self.valid_values = {}
        self.all_values = None
        # load the properties from a file, first decompress the file with gzip and then load the pickle file
        self.properties = None
        self.num_properties = {}
        self.valid_property_map = {}

        # path to the data
        data_path = f'{path}/{db_name}_{property_name}.prop'
        # path to the info file
        info_path = f'{path}/{db_name}_{property_name}.yml'

        # check if the file exists, otherwise raise an error
        if os.path.isfile(data_path) and os.path.isfile(info_path):
            with gzip.open(data_path, 'rb') as f:
                self.properties = pickle.load(f)

            with open(info_path, 'r') as f:
                loaded = yaml.load(f, Loader=yaml.FullLoader)
                self.all_values = loaded['valid_values']
                # convert to list of tuples or single values
                for i, value in enumerate(self.all_values):
                    if type(value) == str:
                        self.all_values[i] = ast.literal_eval(value)
        else:
            raise FileNotFoundError(f'File {data_path} or {info_path} not found')

        for (layer_id, channel_id), values in valid_values.items():
            self.add_properties(layer_id=layer_id, channel_id=channel_id, valid_values=values)

    def add_properties(self, valid_values: List[int], layer_id: int, channel_id: int):
        self.valid_values[(layer_id, channel_id)] = []
        self.valid_property_map[(layer_id, channel_id)] = {}
        # if property name is edge_label_distance, and the valid values is a list of values interpret them as the distances and take all the values from self.all_values with first entry equal to the distance
        if self.name == 'edge_label_distances':
            # check if valid_values is a list of ints
            if type(valid_values[0]) == int:
                tmp_valid_values = []
                for v in self.all_values:
                    if v[0] in valid_values:
                        tmp_valid_values.append(v)
                self.valid_values[(layer_id, channel_id)] = tmp_valid_values
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        elif self.name == 'circle_distances':
            if type(valid_values[0]) == str:
                for v in valid_values:
                    if v == 'no_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'in_circles':
                        for x in self.all_values:
                            if x[1] == 0 and x[2] == 1:
                                self.valid_values[(layer_id, channel_id)].append(x)
                    if v == 'out_circles':
                        for x in self.all_values:
                            if x[1] == 1 and x[2] == 0:
                                self.valid_values[(layer_id, channel_id)].append(x)
            else:
                self.valid_values[(layer_id, channel_id)] = valid_values
        else:
            self.valid_values[(layer_id, channel_id)] = valid_values

        # check if all the valid values are in the valid properties, if not raise an error
        for value in self.valid_values[(layer_id, channel_id)]:
            if value not in self.all_values:
                raise ValueError(f'Property {value} not in valid properties')

        # number of valid properties
        self.num_properties[(layer_id, channel_id)] = len(self.valid_values[(layer_id, channel_id)])
        for i, value in enumerate(self.valid_values[(layer_id, channel_id)]):
            try:
                property_value = int(value)
                self.valid_property_map[(layer_id, channel_id)][property_value] = i
            except:
                # check if the length of the value is 1, if not iterate over the values
                try:
                    len(value[0])
                    for v in value:
                        self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(v)] = i
                except:
                    self.valid_property_map[(layer_id, channel_id)][convert_to_tuple(value)] = i

