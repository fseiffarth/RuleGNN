'''
Created on 15.03.2019

@author:
'''
from pathlib import Path
from typing import Tuple, Optional

import matplotlib
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.init
import time
import numpy as np

from src.utils import GraphData, GraphDrawing
from src.utils.GraphLabels import NodeLabels


def get_label_string(label_dict: dict)->str:
    """
    Method to get the label string. This is used to load the node labels
    """
    label_type = label_dict.get('label_type', None)
    if label_type is None:
        raise ValueError("Label type is not specified")

    if isinstance(label_type, list):
        l_string = ""
        for i, l in enumerate(label_type):
            new_label_dict = label_dict.copy()
            new_label_dict['label_type'] = l
            if i > 0:
                l_string += "_"
            l_string += get_label_string(new_label_dict)
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
        return l_string

    if label_type == "primary":
        l_string = "primary"
        if 'max_labels' in label_dict:
            max_labels = label_dict['max_labels']
            l_string = f"primary_{max_labels}"
    elif label_type == "index":
        l_string = "index"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"index_{max_labels}"
    elif label_type == "wl":
        iterations = label_dict.get('depth', 3)
        l_string = f"wl_{iterations}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "wl_labeled":
        iterations = label_dict.get('depth', 3)
        l_string = f"wl_labeled_{iterations}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "degree":
        l_string = "wl_0"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "simple_cycles":
        if 'max_cycle_length' in label_dict:
            max_cycle_length = label_dict['max_cycle_length']
            l_string = f"simple_cycles_{max_cycle_length}"
        else:
            l_string = "simple_cycles_max"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "induced_cycles":
        if 'max_cycle_length' in label_dict:
            max_cycle_length = label_dict['max_cycle_length']
            l_string = f"induced_cycles_{max_cycle_length}"
        else:
            l_string = "induced_cycles_max"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "cliques":
        l_string = f"cliques"
        if 'max_clique_size' in label_dict:
            max_clique_size = label_dict['max_clique_size']
            l_string = f"cliques_{max_clique_size}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "subgraph":
        l_string = f"subgraph"
        if 'id' in label_dict:
            subgraph_id = label_dict['id']
            l_string = f"{l_string}_{subgraph_id}"
        max_labels = label_dict.get('max_labels', None)
        if max_labels is not None:
            l_string = f"{l_string}_{max_labels}"
    elif label_type == "trivial":
        l_string = "trivial"
    else:
        raise ValueError(f"Layer type {label_type} is not supported")

    return l_string


class LayerHead:
    def __init__(self, info_dict: dict, head_id):
        self.head_id = head_id
        self.label_dict = info_dict.get('labels', None)
        self.property_dict = info_dict.get('properties', None)
        # if head, tail, bias is not specified, set head tail bias to the same value
        self.head_labels = self.label_dict.get('head', None)
        self.head_node_labels = -1
        self.tail_labels = self.label_dict.get('tail', None)
        self.tail_node_labels = -1
        self.bias_labels = self.label_dict.get('bias', None)
        self.bias_node_labels = -1
        self.bias = info_dict.get('bias', False)
        if self.head_labels is None:
            self.head_labels = self.label_dict
        if self.tail_labels is None:
            self.tail_labels = self.head_labels
        if self.bias_labels is None:
            self.bias_labels = self.head_labels


class Layer:
    """
    classdocs for the Layer: This class represents a layer in a RuleGNN
    """

    def __init__(self, layer_dict, layer_id):
        """
        Constructor of the Layer
        :param layer_dict: the dictionary that contains the layer information
        """
        self.layer_type = layer_dict["layer_type"]
        self.layer_dict = layer_dict
        self.layer_heads = []
        self.layer_id = layer_id
        for c_id, head_entry in enumerate(layer_dict.get('heads', [])):
            self.layer_heads.append(LayerHead(head_entry, c_id))

    def get_unique_layer_dicts(self):
        unique_dicts = []
        for head in self.layer_heads:
            if head.head_labels not in unique_dicts:
                unique_dicts.append(head.head_labels)
            if head.tail_labels not in unique_dicts:
                unique_dicts.append(head.tail_labels)
            if head.bias_labels not in unique_dicts:
                unique_dicts.append(head.bias_labels)
        return unique_dicts

    def get_unique_property_dicts(self):
        unique_dicts = []
        for head in self.layer_heads:
            if head.property_dict not in unique_dicts and head.property_dict is not None:
                unique_dicts.append(head.property_dict)
        return unique_dicts

    def get_head_string(self, head_id=0):
        return get_label_string(self.layer_heads[head_id].head_labels)
    def get_tail_string(self, head_id=0):
        return get_label_string(self.layer_heads[head_id].tail_labels)
    def get_bias_string(self, head_id=0):
        return get_label_string(self.layer_heads[head_id].bias_labels)

    def get_layer_label_strings(self)->list[str]:
        label_string_list = set()
        for head in range(len(self.layer_heads)):
            label_string_list.add(get_label_string(self.layer_heads[head].head_labels))
            label_string_list.add(get_label_string(self.layer_heads[head].tail_labels))
            label_string_list.add(get_label_string(self.layer_heads[head].bias_labels))
        return list(label_string_list)

    def num_heads(self):
        return len(self.layer_heads)


class RuleConvolutionLayer(nn.Module):
    """
    classdocs for the GraphConvLayer: This class represents a convolutional layer for a RuleGNN
    """

    def __init__(self, layer_id, seed, parameters, layer: Layer, graph_data: GraphData.RuleGNNDataset, device='cpu'):
        """
        Constructor of the GraphConvLayer
        :param layer_id: the id of the layer
        :param seed: the seed for the random number generator
        :param parameters: the parameters of the experiment
        :param graph_data: the data of the graph dataset
        :param bias: if bias is used in the layer
        :param precision: the precision of the weights, can be torch.float or torch.double
        """
        super(RuleConvolutionLayer, self).__init__()
        # set seed for reproducibility
        torch.manual_seed(seed)
        # id and name of the layer
        self.layer_id = layer_id
        # layer information
        self.layer = layer
        self.name = f"Rule Convolution Layer"
        # get the graph data
        self.graph_data = graph_data
        # get the input features, i.e. the dimension of the input vector
        self.input_feature_dimensions = self.graph_data.num_node_features
        # set the node labels
        self.n_head_labels = []
        self.n_tail_labels = []
        self.n_bias_labels = []
        self.n_properties = []
        self.out_heads = len(layer.layer_heads)
        self.head_strings = []
        self.tail_strings = []
        self.bias_strings = []
        self.property_names = []
        for h_id, head in enumerate(layer.layer_heads):
            self.head_strings.append(layer.get_head_string(h_id))
            self.n_head_labels.append(graph_data.node_labels[self.head_strings[h_id]].num_unique_node_labels)
            self.tail_strings.append(layer.get_tail_string(h_id))
            self.n_tail_labels.append(graph_data.node_labels[self.tail_strings[h_id]].num_unique_node_labels)
            self.bias_strings.append(layer.get_bias_string(h_id))
            self.n_bias_labels.append(graph_data.node_labels[self.bias_strings[h_id]].num_unique_node_labels)
            self.property_names.append(head.property_dict["name"])
            self.n_properties.append(graph_data.properties[self.property_names[h_id]].num_properties[(layer_id,h_id)])


        # head-wise weight num
        self.weight_num = []
        self.bias_num = []

        self.para = parameters  # get the all the parameters of the experiment
        self.bias_list = [head.bias for head in layer.layer_heads]
        self.bias = any(self.bias_list)  # check if bias is used
        self.device = device  # set the device
        self.precision = torch.float # set the precision of the weights
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        # Initialize the current weight matrix, bias vector and feature weight matrix
        self.current_W = torch.Tensor()
        self.current_B = torch.Tensor()
        self.feature_W = torch.Tensor()
        if parameters.run_config.config.get('use_feature_weights', False):
            self.feature_W = nn.Parameter(torch.randn((self.input_feature_dimensions, self.input_feature_dimensions), dtype=self.precision))


        # Determine the number of weights and biases
        # There are two cases assymetric and symmetric, assymetric is the default
        self.skips = [0]
        self.skips_description = [None]
        self.skips_description_text = [None]
        self.weight_distribution = [None] * len(graph_data)
        self.bias_distribution = [None] * len(graph_data)
        for i, head in enumerate(layer.layer_heads):
            valid_property_values = self.graph_data.properties[self.property_names[i]].valid_values[(layer_id, i)]
            # get subdict of valid properties
            # apply the head and tail labels to the subdict
            head_labels = self.graph_data.node_labels[self.head_strings[i]].node_labels
            tail_labels = self.graph_data.node_labels[self.tail_strings[i]].node_labels
            bias_labels = self.graph_data.node_labels[self.bias_strings[i]].node_labels
            for key in valid_property_values:
                property_subdict = self.graph_data.properties[self.property_names[i]].properties[key]
                property_subdict_slices = self.graph_data.properties[self.property_names[i]].properties_slices[key]
                labeled_subdict = property_subdict.detach().clone()
                labeled_subdict[:, 0] = head_labels[property_subdict[:, 0]]
                labeled_subdict[:, 1] = tail_labels[property_subdict[:, 1]]
                # set all indices to -1 where the head or tail label is -1
                invalid_indices = torch.where(torch.logical_or(labeled_subdict[:, 0] == -1, labeled_subdict[:, 1] == -1))[0]
                do_invalid_indices_exist = len(invalid_indices) > 0
                max_first = torch.max(labeled_subdict[:, 0]) + 1
                max_second = torch.max(labeled_subdict[:, 1]) + 1
                labeled_subdict[invalid_indices] = torch.tensor([max_first, max_second])
                # get unique rows of the property subdict together with counts and indices
                _, indices, counts = torch.unique(labeled_subdict, dim=0, return_inverse=True, return_counts=True, sorted=False)
                if do_invalid_indices_exist:
                    counts[-1] = 0
                # set all indices to -1 where the count is smaller than the threshold TODO
                threshold = self.para.run_config.config.get('rule_occurrence_threshold', 1)
                num_weights = len(counts)
                if do_invalid_indices_exist:
                    num_weights -= 1
                if threshold > 1 or do_invalid_indices_exist:
                    # get a bool tensor from indices where the entry is true if the indices entry is in the unique_rows
                    valid_values = torch.where(counts >= threshold)[0]
                    valid_value_dict = {idx: value.item() for idx, value in enumerate(valid_values)}
                    valid_indices_bool = torch.isin(indices, valid_values)
                    valid_indices = torch.where(valid_indices_bool)[0]
                    # relabel indices
                    indices[valid_indices] = torch.tensor([valid_value_dict[idx.item()] for idx in indices[valid_indices]], dtype=torch.int64)
                    num_weights = len(valid_values)
                for idx in range(len(graph_data)):
                    if threshold > 1 or do_invalid_indices_exist:
                        valid_indices_graph = torch.where(valid_indices_bool[property_subdict_slices[idx]:property_subdict_slices[idx+1]])[0] + property_subdict_slices[idx]
                    else:
                        valid_indices_graph = torch.arange(property_subdict_slices[idx], property_subdict_slices[idx+1], dtype=torch.int64)
                    # create new tensor where each row is the concatenation of head_id, property_subdict_row, and indices
                    new_weight_distribution = torch.zeros((len(valid_indices_graph), 4), dtype=torch.int64)
                    new_weight_distribution[:, 0] = i
                    new_weight_distribution[:, 1:3] = property_subdict[valid_indices_graph] - self.graph_data.slices['x'][idx] # check if subtracting is necessary
                    new_weight_distribution[:, 3] = indices[valid_indices_graph] + self.skips[-1]
                    if self.weight_distribution[idx] is None:
                        self.weight_distribution[idx] = new_weight_distribution.detach().clone()
                    else:
                        self.weight_distribution[idx] = torch.cat((self.weight_distribution[idx], new_weight_distribution), dim=0)



                self.skips.append(self.skips[-1] + num_weights)
                self.skips_description.append({'head:': i, 'property': key, 'weights': num_weights})
                self.skips_description_text.append(f"Head {i} Property {key} has {num_weights} different weights")


            self.weight_num.append(self.skips[-1])
            # TODO symmetric case
            if self.bias:
                # Determine the number of different learnable parameters in the bias vector
                self.bias_num.append(self.input_feature_dimensions * self.n_bias_labels[i])
                # Set the bias weights
                _, indices, counts = torch.unique(bias_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
                for idx in range(len(graph_data)):
                    for feature_id in range(self.input_feature_dimensions):
                        new_bias_distribution = torch.zeros((graph_data.num_nodes[idx].item(), 4), dtype=torch.int64)
                        new_bias_distribution[:, 0] = i
                        new_bias_distribution[:, 1] = torch.arange(graph_data.num_nodes[idx].item(), dtype=torch.int64) # alternative torch.arange(start=graph_data.slices['x'][idx], end=graph_data.slices['x'][idx+1], dtype=torch.int64)
                        new_bias_distribution[:, 2] = feature_id
                        new_bias_distribution[:, 3] = indices[graph_data.slices['x'][idx]:graph_data.slices['x'][idx+1]] + feature_id * self.n_bias_labels[i]
                        if self.bias_distribution[idx] is None:
                            self.bias_distribution[idx] = new_bias_distribution.detach().clone()
                        else:
                            self.bias_distribution[idx] = torch.cat((self.bias_distribution[idx], new_bias_distribution), dim=0)

        # Merge the weight distribution of all graphs (creating additionally slicing information)
        self.weight_distribution_slices = torch.tensor([0] + [len(w) for w in self.weight_distribution], dtype=torch.int64).cumsum(dim=0)
        self.weight_distribution = torch.cat([self.weight_distribution[i] for i in range(len(graph_data))], dim=0)
        if self.bias:
            # Merge the bias distribution of all graphs (creating additionally slicing information)
            self.bias_distribution_slices = torch.tensor([0] + [len(b) for b in self.bias_distribution], dtype=torch.int64).cumsum(dim=0)
            self.bias_distribution = torch.cat([self.bias_distribution[i] for i in range(len(graph_data))], dim=0)


        if self.bias:
            #self.bias_map = np.arange(total_bias_num, dtype=np.int64).reshape((self.n_bias_labels, self.input_feature_dimension))
            self.Param_b = self.init_weights(np.sum(self.bias_num), init_type='convolution_bias')
        self.Param_W = self.init_weights(np.sum(self.weight_num), init_type='convolution')


        # TODO add pruning
        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size())

        self.forward_step_time = 0

    def init_weights(self, num_weights, init_type=None):
        weights = nn.Parameter(torch.zeros(num_weights, dtype=self.precision), requires_grad=True)
        weight_init = self.para.run_config.config.get('weight_initialization', None)
        if weight_init is not None:
            weight_initialization = weight_init.get(init_type, None)
            if weight_initialization is not None:
                if weight_initialization.get('type', None) == 'uniform':
                    torch.nn.init.uniform_(weights, a=weight_initialization.get('min', 0.0), b=weight_initialization.get('max', 1.0))
                elif weight_initialization.get('type', None) == 'normal':
                    torch.nn.init.normal_(weights, mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0))
                elif weight_initialization.get('type', None) == 'symmetric_normal':
                    # choose from two normal distributions one with positive and one with negative mean
                    # shuffle the indices
                    weight_arrange = torch.randperm(torch.arange(0, num_weights).size(0))
                    # initialize the weights with indeces in weight_arrange[0:num_weights//2] with positive mean and the rest with negative mean
                    new_weights = torch.zeros(num_weights, dtype=self.precision)
                    new_weights[weight_arrange[0:num_weights//2]] = torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[0:num_weights//2].size(0),), dtype=self.precision)
                    new_weights[weight_arrange[num_weights//2:]] = -torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[num_weights//2:].size(0),), dtype=self.precision)
                    weights = nn.Parameter(new_weights, requires_grad=True)

                elif weight_initialization.get('type', None) == 'constant':
                    torch.nn.init.constant_(weights, weight_initialization.get('value', 0.01))
                elif weight_initialization.get('type', None) == 'lower_upper':
                    # calculate the range for the weights
                    lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
                    weights = nn.Parameter(lower + torch.randn(num_weights, dtype=self.precision) * (upper - lower))
                elif weight_initialization.get('type', None) == 'he':
                    std = np.sqrt(2.0 / num_weights)
                    weights = nn.Parameter(torch.randn(num_weights, dtype=self.precision) * std)

            else:
                raise ValueError(f"Weight initialization type {init_type} is not supported")
        else:
            torch.nn.init.constant_(weights, 0.01)
        return weights

    def set_weights(self, pos):
        input_size = self.graph_data.num_nodes[pos].item()
        # reshape self.current_W to the size of the weight matrix and fill it with minus infinity
        #self.current_W = torch.fill(torch.zeros((self.out_heads, input_size, input_size), dtype=self.precision).to(self.device), float('-inf'))
        self.current_W = torch.zeros((self.out_heads, input_size, input_size), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[self.weight_distribution_slices[pos]:self.weight_distribution_slices[pos+1]]
        if len(weight_distr) != 0:
            # get third column of the weight_distribution: the index of self.Param_W
            param_weights = weight_distr[:, 3]
            matrix_indices = weight_distr[:, 0:3].T
            # set current_W by using the matrix_indices with the values of the Param_W at the indices of param_indices
            self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, param_weights)
        pass

    def set_bias(self, pos):
        input_size = self.graph_data.num_nodes[pos].item()
        self.current_B = torch.zeros((self.out_heads, input_size, self.input_feature_dimensions), dtype=self.precision).to(self.device)
        bias_distr = self.bias_distribution[self.bias_distribution_slices[pos]:self.bias_distribution_slices[pos+1]]
        param_indices = bias_distr[:, 3]
        matrix_indices = bias_distr[:, 0:3].T
        self.current_B[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_b, param_indices)

    def print_layer_info(self):
        print("Layer" + self.__class__.__name__)

    def print_weights(self):
        print("Weights of the Convolution layer")
        string = ""
        for x in self.Param_W:
            string += str(x.data)
        print(string)

    def print_bias(self):
        print("Bias of the Convolution layer")
        for x in self.Param_b:
            print("\t", x.data)

    def print_all(self):
        # print the layer name
        print("Layer: ", self.name)
        print("\tLearnable Weights:")
        # print non-zero/total parameters
        num_params = self.Param_W.numel()
        num_non_zero_params = torch.nonzero(self.Param_W).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        # print relative number of non-zero parameters
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")
        # print the bias parameters
        print("\tLearnable Bias:")
        num_params = self.Param_b.numel()
        num_non_zero_params = torch.nonzero(self.Param_b).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")


    def forward(self, x, pos):
        #x = x.view(-1)
        # print(x.size()[0])
        begin = time.time()
        # set the weights
        self.set_weights(pos)
        if self.bias:
            self.set_bias(pos)
            self.forward_step_time += time.time() - begin
            if self.para.run_config.config.get('degree_matrix', False):
                torch.einsum('cij,jk->cik', torch.diag(self.D[pos]) @ self.current_W @ torch.diag(self.D[pos]), x) + self.current_B
            elif self.para.run_config.config.get('use_in_degrees', False):
                self.in_edges[pos] * torch.einsum('cij,jk->cik', self.current_W, x) + self.current_B
            else:
                x = torch.einsum('cij,jk->cik', self.current_W, x) + self.current_B
        else:
            self.forward_step_time += time.time() - begin
            if self.para.run_config.config.get('degree_matrix', False):
                x = self.in_edges[pos]*torch.einsum('cij,jk->cik', torch.diag(self.D[pos]) @ self.current_W @ torch.diag(self.D[pos]), x)
            elif self.para.run_config.config.get('use_in_degrees', False):
                x = self.in_edges[pos]*torch.einsum('cij,jk->cik', self.current_W, x)
            else:
                x = torch.einsum('cij,jk->cik', self.current_W, x)
                # use row-wise softmax to normalize the weights
                #x = torch.einsum('cij,jk->cik', torch.nn.functional.softmax(self.current_W, dim=2), x)

                # print torch type of current_W and x
        x = x.permute(1, 2, 0)
        # if last dimension is 1, remove it
        if x.size(2) == 1:
            x = x.squeeze(2)
        return x


    def get_weights(self):
        # return the weights as a numpy array
        return np.array(self.Param_W.detach().cpu())

    def get_graph_weights(self, graph_id):
        return self.weight_distribution[self.weight_distribution_slices[graph_id]:self.weight_distribution_slices[graph_id + 1]]

    def get_bias(self):
        if self.bias:
            return np.array(self.Param_b.detach().cpu())
        else:
            return None

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, filter_weights=None, with_graph=True, graph_only=False, draw_bias_labels=False,pos_path:str=''):
        # create graph
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        pos = dict()
        # if pos_path is given and the file exists, load the positions from the file
        if pos_path != '' and Path(pos_path).is_file():
            pos = dict()
            with open(pos_path, 'r') as f:
                # iterate over the lines of the file
                for line in f:
                    # split the line by whitespaces
                    line = line.split()
                    # get the node id and the x and y position
                    pos[int(line[0])] = (float(line[1]), float(line[2]))
        if with_graph or graph_only:
            # draw the graph
            if not Path(pos_path).is_file():
                # if graph is circular use the circular layout
                if graph_drawing[0].draw_type == 'circle':
                    # root node is the one with label 0
                    root_node = None
                    for node in graph.nodes():
                        if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                            root_node = node
                            break
                    # get circular positions around (0,0) starting with the root node at (-400,0)
                    pos[root_node] = (400, 0)
                    angle = 2 * np.pi / (graph.number_of_nodes())
                    # iterate over the neighbors of the root node
                    cur_node = root_node
                    last_node = None
                    counter = 0
                    while len(pos) < graph.number_of_nodes():
                        neighbors = list(graph.neighbors(cur_node))
                        for next_node in neighbors:
                            if next_node != last_node:
                                counter += 1
                                pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                                last_node = cur_node
                                cur_node = next_node
                                break
                elif graph_drawing[0].draw_type == 'kawai':
                    pos = nx.kamada_kawai_layout(graph)
                elif graph_drawing[0].draw_type == 'shell':
                    pos = nx.shell_layout(graph)
                elif graph_drawing[0].draw_type == 'bfs':
                    pos = nx.bfs_layout(graph, 0)
                else:
                    pos = nx.nx_pydot.graphviz_layout(graph)

                # keys to ints
                pos = {int(k): v for k, v in pos.items()}
                # if pos_path is given, save the positions to the file
                if pos_path != '':
                    with open(pos_path, 'w') as f:
                        for key, value in pos.items():
                            f.write(f"{key} {value[0]} {value[1]}\n")
            if graph_only:
                edge_labels = {}
                for (key1, key2, value) in graph.edges(data=True):
                    if "label" in value and len(value["label"]) > 1:
                        edge_labels[(key1, key2)] = int(value["label"][0])
                    else:
                        edge_labels[(key1, key2)] = ""
                nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[0].edge_color,
                                       width=graph_drawing[0].edge_width)
                nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8,
                                             font_color='black')
                # get node colors from the node labels using the plasma colormap


                draw_node_labels = None
                num_unique_node_labels = 0
                if isinstance(self.graph_data.node_labels['primary'], NodeLabels):
                    draw_node_labels = self.graph_data.node_labels['primary'].node_labels
                    num_unique_node_labels = self.graph_data.node_labels['primary'].num_unique_node_labels
                elif isinstance(self.graph_data.node_labels['primary'], torch.Tensor):
                    draw_node_labels = self.graph_data.node_labels['primary']
                    num_unique_node_labels = torch.unique(draw_node_labels).size(0)
                else:
                    raise ValueError("Node labels are not of type NodeLabels or torch.Tensor")

                graph_node_labels = None
                if draw_bias_labels:
                    graph_node_labels = self.graph_data.node_labels[self.bias_strings[head]].node_labels[self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]]
                    num_unique_node_labels = self.graph_data.node_labels[self.bias_strings[head]].num_unique_node_labels
                else:
                    if isinstance(self.graph_data.node_labels['primary'], NodeLabels):
                        graph_node_labels = self.graph_data.node_labels['primary'].node_labels[self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]]
                    elif isinstance(self.graph_data.node_labels['primary'], torch.Tensor):
                        graph_node_labels = self.graph_data.node_labels['primary'][self.graph_data.slices['x'][graph_id]:self.graph_data.slices['x'][graph_id+1]]
                    else:
                        raise ValueError("Node labels are not of type NodeLabels or torch.Tensor")

                cmap = graph_drawing[0].colormap
                norm = matplotlib.colors.Normalize(vmin=0, vmax=num_unique_node_labels)
                node_colors = [cmap(norm(graph_node_labels[node])) for node
                               in graph.nodes()]
                nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                       node_size=graph_drawing[0].node_size)
                #nx.draw_networkx_labels(graph, pos=pos, ax=ax, labels={node: node for node in graph.nodes()}, font_size=8)
                return
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color, width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        all_weights = np.array(self.get_weights())
        bias = self.get_bias()
        weight_distribution = self.get_graph_weights(graph_id)
        param_indices = np.array(weight_distribution[:, 3])
        matrix_indices = np.array(weight_distribution[:, 0:3])
        graph_weights = all_weights[param_indices]

        # sort weights
        if filter_weights is not None:
            sorted_weights = np.sort(np.array(list(set(graph_weights))))
            if filter_weights.get('percentage', None) is not None:
                percentage = filter_weights['percentage']
                lower_bound_weight = sorted_weights[int(len(sorted_weights) * percentage) - 1]
                upper_bound_weight = sorted_weights[int(len(sorted_weights) * (1 - percentage))]
            elif filter_weights.get('absolute', None) is not None:
                absolute = filter_weights['absolute']
                lower_bound_weight = sorted_weights[absolute - 1]
                upper_bound_weight = sorted_weights[-absolute]
            # set all weights smaller than the lower bound and larger than the upper bound to zero
            upper_weights = np.where(graph_weights >= upper_bound_weight, graph_weights, 0)
            lower_weights = np.where(graph_weights <= lower_bound_weight, graph_weights, 0)

            weights = upper_weights + lower_weights
        else:
            weights = np.asarray(graph_weights)

        weight_min = np.min(graph_weights)
        weight_max = np.max(graph_weights)
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        # use seismic colormap with maximum and minimum values from the weight matrix
        cmap = graph_drawing[1].colormap
        # normalize item number values to colormap
        normed_weight = (graph_weights + (-weight_min)) / (weight_max - weight_min)
        weight_colors = cmap(normed_weight)

        if self.bias:
            bias_min = np.min(bias)
            bias_max = np.max(bias)
            bias_max_abs = max(abs(bias_min), abs(bias_max))
            normed_bias = (bias + (-bias_min)) / (bias_max - bias_min)
            bias_colors = cmap(normed_bias)




        # draw the graph
        # if graph is circular use the circular layout
        # draw the graph
        if pos == {}:
            if graph_drawing[0].draw_type == 'circle':
                # root node is the one with label 0
                root_node = None
                for i, node in enumerate(graph.nodes()):
                    if i == 0:
                        print(f"First node: {self.graph_data.node_labels['primary'].node_labels[graph_id][node]}")
                    if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                        root_node = node
                        break
                # get circular positions around (0,0) starting with the root node at (-400,0)
                pos[root_node] = (400, 0)
                angle = 2 * np.pi / (graph.number_of_nodes())
                # iterate over the neighbors of the root node
                cur_node = root_node
                last_node = None
                counter = 0
                while len(pos) < graph.number_of_nodes():
                    neighbors = list(graph.neighbors(cur_node))
                    for next_node in neighbors:
                        if next_node != last_node:
                            counter += 1
                            pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                            last_node = cur_node
                            cur_node = next_node
                            break
            elif graph_drawing[0].draw_type == 'kawai':
                pos = nx.kamada_kawai_layout(graph)
            elif graph_drawing[0].draw_type == 'shell':
                pos = nx.shell_layout(graph)
            elif graph_drawing[0].draw_type == 'bfs':
                pos = nx.bfs_layout(graph,0)
            else:
                pos = nx.nx_pydot.graphviz_layout(graph)
            # keys to ints
            pos = {int(k): v for k, v in pos.items()}
            # if pos_path is given, save the positions to the file
            if pos_path != '':
                with open(pos_path, 'w') as f:
                    for key, value in pos.items():
                        f.write(f"{key} {value[0]} {value[1]}\n")
        # graph to digraph with
        digraph = nx.DiGraph()
        for node in graph.nodes():
            digraph.add_node(node)


        if self.bias:
            node_colors = []
            node_sizes = []
            for node in digraph.nodes():
                node_label = self.graph_data.node_labels[self.bias_strings[head]].node_labels[graph_id][node]
                node_colors.append(bias_colors[node_label])
                node_sizes.append(graph_drawing[1].node_size * abs(bias[node_label]) / bias_max_abs)

            nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)

        edge_widths = []
        for weight_id, entry in enumerate(weight_distribution):
            c = entry[0]
            if c == head:
                i = entry[1]
                j = entry[2]
                if weights[weight_id] != 0:
                    # add edge with weight as data
                    digraph.add_edge(i.item(), j.item(), weight=weight_id)
        curved_edges = [edge for edge in digraph.edges(data=True)]
        curved_edges_colors = []

        for edge in curved_edges:
            curved_edges_colors.append(weight_colors[edge[2]['weight']])
            edge_widths.append(graph_drawing[1].weight_edge_width * abs(weights[edge[2]['weight']]) / weight_max_abs)
        arc_rad = 0.25
        nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                               width=edge_widths,
                               connectionstyle=f'arc3, rad = {arc_rad}', arrows=True, arrowsize=graph_drawing[1].arrow_size, node_size=graph_drawing[1].node_size)




class RuleAggregationLayer(nn.Module):
    '''
    The RuleAggregationLayer class represents the aggregation layer of the RuleGNN
    It gets as input a matrix of size (nodes x graph_data.input_feature_dimension) and returns a matrix of size (output_dimension x graph_data.input_feature_dimension)
    '''
    def __init__(self, layer_id, seed, parameters, layer: Layer, graph_data: GraphData.RuleGNNDataset,
                 out_dim, device='cpu'):

        super(RuleAggregationLayer, self).__init__()
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.name = f"Rule Aggregation Layer"
        self.para = parameters
        # id of the layer
        self.layer_id = layer_id
        # all the layer parameters
        self.layer = layer
        # get the graph data
        self.graph_data = graph_data
        self.precision = torch.float
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double
        self.output_dimension = out_dim
        self.out_heads = len(layer.layer_heads)
        # device
        self.device = device
        self.n_node_labels = []
        self.head_strings = []
        # bias per head
        self.bias_list = [head.bias for head in layer.layer_heads]
        # is there any bias
        self.bias = any(self.bias_list)
        for i, head in enumerate(layer.layer_heads):
            self.head_strings.append(layer.get_head_string(i))
            self.n_node_labels.append(graph_data.node_labels[self.head_strings[i]].num_unique_node_labels)

        self.input_feature_dimension = self.graph_data.num_node_features

        self.weight_num = np.sum(self.n_node_labels) * out_dim
        #self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape((self.heads, out_dim, n_node_labels))
        self.current_W = torch.Tensor()


        self.weight_distribution = [None] * len(graph_data)
        for i, head in enumerate(layer.layer_heads):
            node_labels = self.graph_data.node_labels[self.head_strings[i]].node_labels
            # Set the bias weights
            _, indices, counts = torch.unique(node_labels, dim=0, return_inverse=True, return_counts=True, sorted=False)
            for idx in range(len(graph_data)):
                for out_dim_id in range(out_dim):
                    new_weight_distribution = torch.zeros((graph_data.num_nodes[idx].item(), 4), dtype=torch.int64)
                    new_weight_distribution[:, 0] = i
                    new_weight_distribution[:, 1] = out_dim_id
                    new_weight_distribution[:, 2] = torch.arange(graph_data.num_nodes[idx].item()) # torch.arange(start=graph_data.slices['x'][idx], end=graph_data.slices['x'][idx+1], dtype=torch.int64)
                    new_weight_distribution[:, 3] = indices[graph_data.slices['x'][idx]:graph_data.slices['x'][idx+1]] + out_dim_id * self.n_node_labels[i]
                    if self.weight_distribution[idx] is None:
                        self.weight_distribution[idx] = new_weight_distribution.detach().clone()
                    else:
                        self.weight_distribution[idx] = torch.cat((self.weight_distribution[idx], new_weight_distribution), dim=0)


        # merge the bias distribution of all graphs (creating additionally slicing information)
        self.weight_distribution_slices = torch.tensor([0] + [len(w) for w in self.weight_distribution], dtype=torch.int64).cumsum(dim=0)
        self.weight_distribution = torch.cat([self.weight_distribution[i] for i in range(len(graph_data))], dim=0)

        self.Param_W = self.init_weights(self.weight_num, init_type='aggregation')


        if self.bias:
            self.Param_b = self.init_weights(shape=(self.out_heads, out_dim, self.input_feature_dimension), init_type='aggregation_bias')
        self.forward_step_time = 0



        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size(), requires_grad=False)

    def init_weights(self, shape, init_type=None):
        num_weights = np.prod(shape)
        weights = nn.Parameter(torch.zeros(shape, dtype=self.precision), requires_grad=True)
        weight_init = self.para.run_config.config.get('weight_initialization', None)
        if weight_init is not None:
            weight_initialization = weight_init.get(init_type, None)
            if weight_initialization is not None:
                if weight_initialization.get('type', None) == 'uniform':
                    torch.nn.init.uniform_(weights, a=weight_initialization.get('min', 0.0), b=weight_initialization.get('max', 1.0))
                elif weight_initialization.get('type', None) == 'normal':
                    torch.nn.init.normal_(weights, mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0))
                elif weight_initialization.get('type', None) == 'symmetric_normal':
                    # choose from two normal distributions one with positive and one with negative mean
                    # shuffle the indices
                    weight_arrange = torch.randperm(torch.arange(0, num_weights).size(0))
                    # initialize the weights with indeces in weight_arrange[0:num_weights//2] with positive mean and the rest with negative mean
                    new_weights = torch.zeros(num_weights, dtype=self.precision)
                    new_weights[weight_arrange[0:num_weights//2]] = torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[0:num_weights//2].size(0),), dtype=self.precision)
                    new_weights[weight_arrange[num_weights//2:]] = -torch.normal(mean=weight_initialization.get('mean', 0.0), std=weight_initialization.get('std', 1.0), size=(weight_arrange[num_weights//2:].size(0),), dtype=self.precision)
                    # reshape new_weights to the shape of the weights
                    new_weights = new_weights.reshape(shape)
                    weights = nn.Parameter(new_weights, requires_grad=True)
                elif weight_initialization.get('type', None) == 'constant':
                    torch.nn.init.constant_(weights, weight_initialization.get('value', 0.01))
                elif weight_initialization.get('type', None) == 'lower_upper':
                    # calculate the range for the weights
                    lower, upper = -(1.0 / np.sqrt(num_weights)), (1.0 / np.sqrt(num_weights))
                    weights = nn.Parameter(lower + torch.randn(self.weight_num, dtype=self.precision) * (upper - lower))
            else:
                torch.nn.init.constant_(weights, 0.01)
        else:
            torch.nn.init.constant_(weights, 0.01)
        return weights

    def set_weights(self, pos):
        input_size = self.graph_data.num_nodes[pos].item()
        self.current_W = torch.zeros((self.out_heads, self.output_dimension, input_size), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[self.weight_distribution_slices[pos]:self.weight_distribution_slices[pos+1]]
        param_indices = weight_distr[:, 3]
        matrix_indices = weight_distr[:, 0:3].T
        self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, param_indices)
        # divide the weights by the number of nodes in the graph
        self.current_W = self.current_W / input_size
        pass

    def print_weights(self):
        print("Weights of the Resize layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Resize layer")
        for x in self.Param_b:
            print("\t", x.data)


    def print_all(self):
        # print the layer name
        print("Layer: ", self.name)
        print("\tLearnable Weights:")
        # print non-zero/total parameters
        num_params = self.Param_W.numel()
        num_non_zero_params = torch.nonzero(self.Param_W).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        # print relative number of non-zero parameters
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")
        # print the bias parameters
        print("\tLearnable Bias:")
        num_params = self.Param_b.numel()
        num_non_zero_params = torch.nonzero(self.Param_b).size(0)
        print(f"\t\tNon-zero parameters: {num_non_zero_params}/{num_params}")
        print(f"\t\tRelative non-zero parameters: {num_non_zero_params / num_params * 100:.2f}%")

    def forward(self, x, pos):
        #x = x.view(-1)
        begin = time.time()
        self.set_weights(pos)

        self.forward_step_time += time.time() - begin

        if self.bias:
            x = torch.einsum('cij,jk->cik', self.current_W, x) + self.Param_b
        else:
            x = torch.einsum('cij,jk->cik', self.current_W, x)
        # flatten the output
        return x.flatten()
        # if self.bias:
        #     return torch.mv(self.weight_matrices[pos], x) + self.Param_b.to("cpu")
        # else:
        #     return torch.mv(self.weight_matrices[pos], x)

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b[0]]

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], head=0, out_dimension=0, with_graph=True, graph_only=False):
        # create graph
        graph = self.graph_data.create_nx_graph(graph_id, directed=False)
        if with_graph or graph_only:
            # draw the graph
            # root node is the one with label 0
            root_node = None
            for node in graph.nodes():
                if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                    root_node = node
                    break

            # if graph is circular use the circular layout
            pos = dict()
            if graph_drawing[0].draw_type == 'circle':
                # get circular positions around (0,0) starting with the root node at (-400,0)
                pos[root_node] = (400, 0)
                angle = 2 * np.pi / (graph.number_of_nodes())
                # iterate over the neighbors of the root node
                cur_node = root_node
                last_node = None
                counter = 0
                while len(pos) < graph.number_of_nodes():
                    neighbors = list(graph.neighbors(cur_node))
                    for next_node in neighbors:
                        if next_node != last_node:
                            counter += 1
                            pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                            last_node = cur_node
                            cur_node = next_node
                            break
            elif graph_drawing[0].draw_type == 'kawai':
                pos = nx.kamada_kawai_layout(graph)
            elif graph_drawing[0].draw_type == 'shell':
                pos = nx.shell_layout(graph)
            elif graph_drawing[0].draw_type == 'bfs':
                pos = nx.bfs_layout(graph, 0)
            else:
                pos = nx.nx_pydot.graphviz_layout(graph)
            # keys to ints
            pos = {int(k): v for k, v in pos.items()}
            if graph_only:
                edge_labels = {}
                for (key1, key2, value) in graph.edges(data=True):
                    if "label" in value and len(value["label"]) > 1:
                        edge_labels[(key1, key2)] = int(value["label"][0])
                    else:
                        edge_labels[(key1, key2)] = ""
                nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[0].edge_color,
                                       width=graph_drawing[0].edge_width)
                nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8,
                                             font_color='black')
                # get node colors from the node labels using the plasma colormap
                cmap = graph_drawing[0].colormap
                norm = matplotlib.colors.Normalize(vmin=0,
                                                   vmax=self.graph_data.node_labels['primary'].num_unique_node_labels)
                node_colors = [cmap(norm(self.graph_data.node_labels['primary'].node_labels[graph_id][node])) for node
                               in graph.nodes()]
                nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors,
                                       node_size=graph_drawing[0].node_size)
                return
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color, width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        all_weights = np.array(self.get_weights())
        bias = self.get_bias()
        graph = self.graph_data.graphs[graph_id]
        weight_distribution = self.weight_distribution[graph_id]
        param_indices = np.array(weight_distribution[:, 3])
        matrix_indices = np.array(weight_distribution[:, 0:3])
        graph_weights = all_weights[param_indices]

        weight_min = np.min(graph_weights)
        weight_max = np.max(graph_weights)
        weight_max_abs = max(abs(weight_min), abs(weight_max))
        bias_min = np.min(bias)
        bias_max = np.max(bias)
        bias_max_abs = max(abs(bias_min), abs(bias_max))

        # use seismic colormap with maximum and minimum values from the weight matrix
        cmap = graph_drawing[1].colormap
        # normalize item number values to colormap
        normed_weight = (graph_weights + (-weight_min)) / (weight_max - weight_min)
        weight_colors = cmap(normed_weight)
        normed_bias = (bias + (-bias_min)) / (bias_max - bias_min)
        bias_colors = cmap(normed_bias)

        # draw the graph
        # if graph is circular use the circular layout
        pos = dict()
        if graph_drawing[0].draw_type == 'circle':
            # root node is the one with label 0
            root_node = None
            for i, node in enumerate(graph.nodes()):
                if i == 0:
                    print(f"First node: {self.graph_data.node_labels['primary'].node_labels[graph_id][node]}")
                if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                    root_node = node
                    break
            # get circular positions around (0,0) starting with the root node at (-400,0)
            pos[root_node] = (400, 0)
            angle = 2 * np.pi / (graph.number_of_nodes())
            # iterate over the neighbors of the root node
            cur_node = root_node
            last_node = None
            counter = 0
            while len(pos) < graph.number_of_nodes():
                neighbors = list(graph.neighbors(cur_node))
                for next_node in neighbors:
                    if next_node != last_node:
                        counter += 1
                        pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                        last_node = cur_node
                        cur_node = next_node
                        break
        elif graph_drawing[0].draw_type == 'kawai':
            pos = nx.kamada_kawai_layout(graph)
        elif graph_drawing[0].draw_type == 'shell':
            pos = nx.shell_layout(graph)
        elif graph_drawing[0].draw_type == 'bfs':
            pos = nx.bfs_layout(graph,0)
        else:
            pos = nx.nx_pydot.graphviz_layout(graph)
        # keys to ints
        pos = {int(k): v for k, v in pos.items()}
        # graph to digraph with
        digraph = nx.DiGraph()
        for node in graph.nodes():
            digraph.add_node(node)



        node_colors = []
        node_sizes = []
        for i, index in enumerate(weight_distribution):
            c = index[0]
            o_dimension = index[1]
            node_idx = index[2]
            weight = index[3]
            if c == head and o_dimension == out_dimension:
                node_colors.append(weight_colors[i])
                node_sizes.append(graph_drawing[1].node_size * abs(graph_weights[i]) / weight_max_abs)

        nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)


class RuleGNNConcatenate(nn.Module):
    """
    Wrapper class for a linear layer that ignores the pos argument
    """
    def __init__(self, in_features, bias=True):
        super(RuleGNNConcatenate, self).__init__()
        self.linear = nn.Linear(in_features, 1, bias=bias)

    def forward(self, x, pos):
        x = self.linear(x)
        # remove last dimension
        return x.squeeze(-1)


class RuleGNNLinear(nn.Module):
    """
    Wrapper class for a linear layer that ignores the pos argument
    """

    def __init__(self, in_features, out_features, bias=True):
        super(RuleGNNLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, pos):
        return self.linear(x)
