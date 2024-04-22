'''
Created on 15.03.2019

@author: florian
'''
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init
import torch.optim as optim
from torch.autograd import Variable
import time
import networkx as nx
import numpy as np
from numpy import int64
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process
import ReadWriteGraphs.GraphFunctions as gf
from GraphData import GraphData


class Layer:
    def __init__(self, layer_dict):
        self.layer_type = layer_dict["layer_type"]
        self.node_labels = -1
        self.layer_dict = layer_dict
        if 'max_node_labels' in layer_dict:
            self.node_labels = layer_dict["max_node_labels"]
        if 'distances' in layer_dict:
            self.distances = layer_dict["distances"]
        else:
            self.distances = None


    def get_layer_string(self):
        l_string = ""
        if self.layer_type == "primary":
            l_string = "primary"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"primary_{max_node_labels}"
        elif self.layer_type == "wl":
            if 'wl_iterations' in self.layer_dict:
                iterations = self.layer_dict['wl_iterations']
                l_string = f"wl_{iterations}"
            else:
                l_string = "wl_max"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "simple_cycles":
            if 'max_cycle_length' in self.layer_dict:
                max_cycle_length = self.layer_dict['max_cycle_length']
                l_string = f"simple_cycles_{max_cycle_length}"
            else:
                l_string = "simple_cycles_max"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "induced_cycles":
            if 'max_cycle_length' in self.layer_dict:
                max_cycle_length = self.layer_dict['max_cycle_length']
                l_string = f"induced_cycles_{max_cycle_length}"
            else:
                l_string = "induced_cycles_max"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "cliques":
            l_string = f"cliques"
            if 'max_clique_size' in self.layer_dict:
                max_clique_size = self.layer_dict['max_clique_size']
                l_string = f"cliques_{max_clique_size}"
        elif self.layer_type == "subgraph":
            l_string = f"subgraph"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"subgraph_{max_node_labels}"

        return l_string


def reshape_indices(a, b):
    dict = {}
    ita = np.nditer(a, flags=['multi_index'])
    itb = np.nditer(b, flags=['multi_index'])
    while not ita.finished:
        dict[ita.multi_index] = itb.multi_index
        ita.iternext()
        itb.iternext()

    return dict


class GraphConvLayer(nn.Module):

    def __init__(self, layer_id, seed, graph_data: GraphData.GraphData, w_distribution_rule, bias_distribution_rule,
                 in_features, node_labels, n_kernels=1, bias=True, print_layer_init=False, save_weights=False, *args,
                 **kwargs):
        super(GraphConvLayer, self).__init__()

        self.layer_id = layer_id
        # get the graph data
        self.graph_data = graph_data
        # get the input features, i.e. the dimension of the input vector
        self.in_features = in_features
        # set the node labels
        self.node_labels = graph_data.node_labels[node_labels]
        n_node_labels = self.node_labels.num_unique_node_labels
        self.edge_labels = graph_data.edge_labels['primary']
        n_edge_labels = self.edge_labels.num_unique_edge_labels
        # get the number of considered node labels
        self.n_node_labels = n_node_labels
        # get the number of considered edge labels
        self.n_edge_labels = n_edge_labels
        # get the number of considered kernels
        self.n_kernels = n_kernels
        self.n_extra_dim = 1
        self.extra_dim_map = {}

        self.args = args
        self.kwargs = kwargs

        # Extra features
        if "n_max_degree" in kwargs:
            self.n_extra_dim = len(kwargs["degrees"])
            for i, x in enumerate(kwargs["degrees"], 0):
                self.extra_dim_map[x] = i
        elif "distances" in kwargs:
            self.distance_list = True
            self.n_extra_dim = len(kwargs["distances"])
            for i, x in enumerate(kwargs["distances"], 0):
                self.extra_dim_map[x] = i
            self.n_edge_labels = 1
            self.args = self.distance_list
        elif "cycle_list" and "cycle_lengths" in kwargs:
            self.cycle_list = kwargs["cycle_list"]
            self.n_extra_dim = len(kwargs["cycle_lengths"])
            for i, x in enumerate(kwargs["cycle_lengths"], 0):
                self.extra_dim_map[x] = i
            self.n_edge_labels = 1
            self.args = self.cycle_list
        else:
            self.extra_dim_map = {0: 0}

        # Determine the number of weights and biases
        self.weight_num = self.in_features * self.in_features * self.n_kernels * self.n_node_labels * self.n_node_labels * self.n_edge_labels * self.n_extra_dim
        self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
            (self.in_features, self.in_features, self.n_kernels, self.n_node_labels, self.n_node_labels,
             self.n_edge_labels, self.n_extra_dim))

        self.bias_num = self.in_features * self.n_kernels * self.n_node_labels
        self.bias_map = np.arange(self.bias_num, dtype=np.int64).reshape(
            (self.in_features, self.n_kernels, self.n_node_labels))

        # Get the rules for the weight and bias distribution
        self.w_distribution_rule = w_distribution_rule
        self.bias_distribution_rule = bias_distribution_rule

        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.Param_W = nn.ParameterList(
            [nn.Parameter(lower + torch.randn(1, dtype=torch.double) * (upper - lower)) for _ in
             range(0, self.weight_num)])
        self.Param_b = nn.ParameterList(
            [nn.Parameter(lower + torch.randn(1, dtype=torch.double) * (upper - lower)) for _ in
             range(0, self.bias_num)])
        self.bias = bias

        # Initialize the current weight matrix and bias vector
        self.current_W = torch.Tensor()
        self.current_B = torch.Tensor()

        self.name = "WL_Layer"

        def valid_node_label(n_label):
            if 0 <= n_label < self.n_node_labels:
                return True
            else:
                return False

        def valid_edge_label(e_label, n1=0, n2=1):
            if 0 <= e_label < self.n_edge_labels:
                return True
            elif n1 == n2 and 0 <= e_label < self.n_edge_labels + 1:
                return True
            else:
                return False

        def valid_extra_dim(extra_dim):
            if extra_dim in self.extra_dim_map:
                return True
            else:
                return False

        # Set the distribution for each graph
        self.weight_distribution = []
        #self.weight_index_list = []
        #self.weight_pos_list = []
        # Set the bias distribution for each graph
        self.bias_distribution = []

        self.weight_matrices = []
        self.bias_weights = []

        for graph_id, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or graph_id % (
                    self.graph_data.num_graphs // 10) == 0) and print_layer_init:
                print("GraphConvLayerInitWeights: ", str(int(graph_id / self.graph_data.num_graphs * 100)), "%")

            node_number = graph.number_of_nodes()
            graph_weight_pos_distribution = np.zeros((1, 3), np.dtype(np.int16))

            input_size = node_number * self.in_features
            weight_entry_num = 0

            in_high_dim = np.zeros(
                shape=(self.in_features, self.in_features, self.n_kernels, node_number, node_number))
            out_low_dim = np.zeros(shape=(input_size, input_size * self.n_kernels))
            index_map = reshape_indices(in_high_dim, out_low_dim)

            #flatten_indices = reshape_indices(out_low_dim, np.zeros(input_size * input_size * self.n_kernels))
            #weight_indices = torch.zeros(1, dtype=torch.int64)
            #weight_pos_tensor = torch.zeros(1, dtype=torch.int64)

            for i1 in range(0, self.in_features):
                for i2 in range(0, self.in_features):
                    for k in range(0, self.n_kernels):
                        for n1 in range(0, node_number):
                            if self.distance_list:
                                # iterate over the distance list until the maximum distance is reached
                                for n2, distance in self.graph_data.distance_list[graph_id][n1].items():
                                    n1_label, n2_label, e_label, extra_dim = self.w_distribution_rule(n1, n2,
                                                                                                      self.graph_data,
                                                                                                      self.node_labels,
                                                                                                      graph_id)
                                    if valid_node_label(int(n1_label)) and valid_node_label(
                                            int(n2_label)) and valid_edge_label(int(e_label)) and valid_extra_dim(
                                        extra_dim):
                                        # position of the weight in the Parameter list
                                        weight_pos = \
                                            self.weight_map[i1][i2][k][int(n1_label)][int(n2_label)][int(e_label)][
                                                self.extra_dim_map[extra_dim]]

                                        # position of the weight in the weight matrix
                                        row_index = index_map[(i1, i2, k, n1, n2)][0]
                                        col_index = index_map[(i1, i2, k, n1, n2)][1]

                                        if weight_entry_num == 0:
                                            graph_weight_pos_distribution[weight_entry_num, 0] = row_index
                                            graph_weight_pos_distribution[weight_entry_num, 1] = col_index
                                            graph_weight_pos_distribution[weight_entry_num, 2] = weight_pos

                                            #weight_indices[0] = flatten_indices[row_index, col_index][0]
                                            #weight_pos_tensor[0] = np.int64(weight_pos).item()
                                        else:
                                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution,
                                                                                      [
                                                                                          [row_index, col_index,
                                                                                           weight_pos]], axis=0)
                                            #weight_indices = torch.cat((weight_indices,torch.tensor([flatten_indices[row_index, col_index][0]])))
                                            #weight_pos_tensor = torch.cat((weight_pos_tensor, torch.tensor([np.int64(weight_pos).item()])))
                                        weight_entry_num += 1
                            else:
                                for n2 in range(0, node_number):
                                    n1_label, n2_label, e_label, extra_dim = self.w_distribution_rule(n1, n2,
                                                                                                      self.graph_data,
                                                                                                      self.node_labels,
                                                                                                      graph_id)
                                    if valid_node_label(int(n1_label)) and valid_node_label(
                                            int(n2_label)) and valid_edge_label(int(e_label)) and valid_extra_dim(
                                        extra_dim):
                                        # position of the weight in the Parameter list
                                        weight_pos = \
                                        self.weight_map[i1][i2][k][int(n1_label)][int(n2_label)][int(e_label)][
                                            self.extra_dim_map[extra_dim]]

                                        # position of the weight in the weight matrix
                                        row_index = index_map[(i1, i2, k, n1, n2)][0]
                                        col_index = index_map[(i1, i2, k, n1, n2)][1]

                                        if weight_entry_num == 0:
                                            graph_weight_pos_distribution[weight_entry_num, 0] = row_index
                                            graph_weight_pos_distribution[weight_entry_num, 1] = col_index
                                            graph_weight_pos_distribution[weight_entry_num, 2] = weight_pos

                                            #weight_indices[0] = flatten_indices[row_index, col_index][0]
                                            #weight_pos_tensor[0] = np.int64(weight_pos).item()
                                        else:
                                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [
                                                [row_index, col_index,
                                                 weight_pos]], axis=0)
                                            #weight_indices = torch.cat((weight_indices, torch.tensor([flatten_indices[row_index, col_index][0]])))
                                            #weight_pos_tensor = torch.cat(weight_pos_tensor, torch.tensor([np.int64(weight_pos).item()])))
                                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            #self.weight_index_list.append(weight_indices)
            #self.weight_pos_list.append(weight_pos_tensor)

            if save_weights:
                parameterMatrix = np.full((input_size, input_size * self.n_kernels), 0, dtype=np.int16)
                self.weight_matrices.append(torch.zeros((input_size, input_size * self.n_kernels), dtype=torch.double))
                for entry in graph_weight_pos_distribution:
                    self.weight_matrices[-1][entry[0]][entry[1]] = self.Param_W[entry[2]]
                    parameterMatrix[entry[0]][entry[1]] = entry[2] + 1
                    np.savetxt(
                        f"Results/{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt",
                        parameterMatrix, delimiter=';', fmt='%i')

            # print(row_array, col_array, data_array)
            # self.sparse_weight_data.append(data_array)
            # self.sparse_weight_row_col.append(torch.cat((row_array, col_array), 0))
            # print(self.sparse_weight_data, self.sparse_weight_row_col)

            graph_bias_pos_distribution = np.zeros((1, 2), np.dtype(np.int16))
            out_size = node_number * self.in_features * self.n_kernels
            bias = torch.zeros((out_size), dtype=torch.double)

            in_high_dim = np.zeros(shape=(self.in_features, self.n_kernels, node_number))
            out_low_dim = np.zeros(shape=(out_size,))
            index_map = reshape_indices(in_high_dim, out_low_dim)
            bias_entry_num = 0
            for i1 in range(0, self.in_features):
                for k in range(0, self.n_kernels):
                    for n1 in range(0, node_number):
                        n1_label = self.bias_distribution_rule(n1, self.node_labels, graph_id)
                        if valid_node_label(int(n1_label)):
                            weight_pos = self.bias_map[i1][k][int(n1_label)]
                            if bias_entry_num == 0:
                                graph_bias_pos_distribution[bias_entry_num][0] = index_map[(i1, k, n1)][0]
                                graph_bias_pos_distribution[bias_entry_num][1] = weight_pos
                            else:
                                graph_bias_pos_distribution = np.append(graph_bias_pos_distribution, [
                                    [index_map[(i1, k, n1)][0], weight_pos]], axis=0)
                            bias_entry_num += 1
            self.bias_distribution.append(graph_bias_pos_distribution)

            if save_weights:
                self.bias_weights.append(bias)
                for entry in graph_bias_pos_distribution:
                    self.bias_weights[-1][entry[0]] = self.Param_b[entry[1]]

        self.forward_step_time = 0

    def set_weights(self, input_size, pos):
        # print(self.sparse_weight_row_col[pos])
        # print(self.sparse_weight_data[pos])
        self.current_W = torch.zeros((input_size, input_size * self.n_kernels), dtype=torch.double)
        weight_distr = self.weight_distribution[pos]
        for entry in weight_distr:
            self.current_W[entry[0], entry[1]] = self.Param_W[entry[2]]
        # W.put_(self.weight_index_list[pos], self.Param_W[self.weight_pos_list[pos]])

        # return self.weight_matrices[pos]

    def set_weights_(self, input_size, pos):
        # print(self.sparse_weight_row_col[pos])
        # print(self.sparse_weight_data[pos])
        current_W = torch.zeros((input_size, input_size * self.n_kernels), dtype=torch.double)
        weight_distr = self.weight_distribution[pos]
        weight_distr = torch.tensor(weight_distr).T
        ind = torch.stack(tuple(weight_distr[0:2]))
        unraveled = torch.tensor(current_W.stride()) @ ind.flatten(1)
        current_W = current_W.flatten()
        current_W[unraveled] = torch.tensor(self.Param_W)[weight_distr[2]]
        # current_W[torch.unravel_index(weight_distr[0:2],shape=(input_size,  input_size * self.n_kernels))] = self.Param_W[weight_distr[2]]
        self.current_W = current_W.reshape(input_size, input_size * self.n_kernels)
        print(self.current_W)
        raise Exception()
        # for entry in weight_distr:
        #    self.current_W[entry[0], entry[1]] = self.Param_W[entry[2]]
        # W.put_(self.weight_index_list[pos], self.Param_W[self.weight_pos_list[pos]])

        # return self.weight_matrices[pos]

    def set_bias(self, input_size, pos):
        self.current_B = torch.zeros((input_size * self.n_kernels), dtype=torch.double)
        bias_distr = self.bias_distribution[pos]
        for entry in bias_distr:
            self.current_B[entry[0]] = self.Param_b[entry[1]]

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

    def forward(self, x, pos):
        x = x.view(-1)
        # print(x.size()[0])
        begin = time.time()
        # set the weights
        self.set_weights(x.size()[0], pos)
        self.set_bias(x.size()[0], pos)
        self.forward_step_time += time.time() - begin
        if self.bias:
            return torch.matmul(self.current_W, x) + self.current_B
        else:
            return torch.mv(self.current_W, x)

        # if self.bias:
        #     return torch.matmul(self.weight_matrices[pos], x) + self.bias_weights[pos]
        # else:
        #     return torch.mv(self.weight_matrices[pos], x)

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b]


class GraphConvDistanceLayer(nn.Module):

    def __init__(self, graph_data: GraphData.GraphData, distance_list, max_distance, w_distribution_rule,
                 bias_distribution_rule,
                 in_features, out_features, n_node_labels, n_kernels=1, bias=True):
        super(GraphConvDistanceLayer, self).__init__()

        self.graph_data = graph_data
        self.distance_list = distance_list
        self.max_distance = max_distance

        self.in_features = in_features
        self.out_features = out_features
        self.n_node_labels = n_node_labels
        self.n_kernels = n_kernels

        self.weight_number = in_features * n_node_labels * n_node_labels * max_distance * n_kernels
        self.bias_weight_number = in_features * n_node_labels * n_kernels

        self.w_distribution_rule = w_distribution_rule
        self.bias_distribution_rule = bias_distribution_rule

        self.Param_W = nn.ParameterList([])
        self.Param_b = nn.ParameterList([])
        self.bias = bias

        for i in range(0, self.weight_number):
            self.Param_W.append(nn.Parameter(torch.tensor(np.random.randn(1))))

        if self.bias:
            for j in range(0, self.bias_weight_number):
                self.Param_b.append(nn.Parameter(torch.randn(1)))

        # Set the distribution for each graph
        self.weight_distribution = []
        for counter, graph in enumerate(self.graph_data.graphs, 0):
            if self.graph_data.num_graphs < 10 or counter % (self.graph_data.num_graphs // 10) == 0:
                print("GraphConvLayerInitWeights: ", str(int(counter / self.graph_data.num_graphs * 100)), "%")
            dist_dat = self.distance_list[counter]
            graph_weight_pos_distribution = np.zeros((1, 3), np.dtype(np.int16))
            input_size = graph.number_of_nodes() * in_features
            weight_entry_num = 0
            W = torch.full((input_size, input_size * self.n_kernels), -1)
            for i in range(0, input_size):
                for j in range(0, input_size * self.n_kernels):
                    number = self.w_distribution_rule(i, j, self.in_features, self.n_node_labels, self.max_distance,
                                                      self.n_kernels, graph, dist_dat)
                    if number >= 0 and number < len(self.Param_W):
                        if weight_entry_num == 0:
                            graph_weight_pos_distribution[weight_entry_num][0] = i
                            graph_weight_pos_distribution[weight_entry_num][1] = j
                            graph_weight_pos_distribution[weight_entry_num][2] = number
                        else:
                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [[i, j, number]],
                                                                      axis=0)
                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            # print(graph_weight_pos_distribution)

        # Set the bias distribution for each graph
        self.bias_distribution = []
        for counter, graph in enumerate(self.graph_data.graphs, 0):
            if self.graph_data.num_graphs < 10 or counter % (self.graph_data.num_graphs // 10) == 0:
                print("GraphConvLayerInitBias: ", str(int(counter / self.graph_data.num_graphs * 100)), "%")
            graph_bias_pos_distribution = np.zeros((1, 2), np.dtype(np.int16))
            input_size = graph.number_of_nodes() * in_features
            bias_entry_num = 0
            bias = torch.full((input_size * self.n_kernels,), -1)

            # set bias weights
            bias_size = bias.size()[0]
            for i in range(0, bias_size):
                number = self.bias_distribution_rule(i, self.in_features, self.n_node_labels, self.n_kernels, graph)
                if number >= 0 and number < len(self.Param_b):
                    if bias_entry_num == 0:
                        graph_bias_pos_distribution[bias_entry_num][0] = i
                        graph_bias_pos_distribution[bias_entry_num][1] = number
                    else:
                        graph_bias_pos_distribution = np.append(graph_bias_pos_distribution, [[i, number]], axis=0)
                    bias_entry_num += 1

            self.bias_distribution.append(graph_bias_pos_distribution)
            # print(graph_bias_pos_distribution)

        self.forward_step_time = 0

    # set weights
    def set_weights(self, input_size, pos):
        W = torch.zeros((input_size, input_size * self.n_kernels))
        weight_distr = self.weight_distribution[pos]
        for entry in weight_distr:
            W[entry[0], entry[1]] = self.Param_W[entry[2]]
        return W

    # set bias weights
    def set_bias_weights(self, input_size, pos):
        bias = torch.zeros((input_size * self.n_kernels))
        bias_distr = self.bias_distribution[pos]
        for entry in bias_distr:
            bias[entry[0]] = self.Param_b[entry[1]]
        return bias

    def print_weights(self):
        print("Weights of the Convolution layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Convolution layer")
        for x in self.Param_b:
            print("\t", x.data)

    def forward(self, x, pos):
        x = x.view(-1).to("cpu")
        begin = time.time()
        """
        p = Process(target=set_weights())
        p.start()
        p.join()
        """
        W = self.set_weights(x.size()[0], pos).to("cpu")
        bias = self.set_bias_weights(x.size()[0], pos).to("cpu")

        self.forward_step_time += time.time() - begin

        # print(W)
        if self.bias:
            return torch.mv(W, x) + bias
        else:
            return torch.mv(W, x)


class GraphCycleLayer(nn.Module):

    def __init__(self, graph_data: GraphData.GraphData, cycle_list, max_cycle_length, w_distribution_rule,
                 bias_distribution_rule,
                 in_features, out_features, n_node_labels, n_kernels=1, bias=True):
        super(GraphCycleLayer, self).__init__()

        self.graph_data = graph_data
        self.cycle_list = cycle_list
        self.max_cycle_length = max_cycle_length

        self.in_features = in_features
        self.out_features = out_features
        self.n_node_labels = n_node_labels
        self.n_kernels = n_kernels

        self.weight_number = in_features * n_node_labels * n_node_labels * max_cycle_length * n_kernels
        self.bias_weight_number = in_features * n_node_labels * n_kernels

        self.w_distribution_rule = w_distribution_rule
        self.bias_distribution_rule = bias_distribution_rule

        self.Param_W = nn.ParameterList([])
        self.Param_b = nn.ParameterList([])
        self.bias = bias

        for i in range(0, self.weight_number):
            self.Param_W.append(nn.Parameter(torch.tensor(np.random.randn(1))))

        if self.bias:
            for j in range(0, self.bias_weight_number):
                self.Param_b.append(nn.Parameter(torch.randn(1)))

        # Set the distribution for each graph
        self.weight_distribution = []
        for counter, graph in enumerate(self.graph_data.graphs, 0):
            if self.graph_data.num_graphs < 10 or counter % (self.graph_data.num_graphs // 10) == 0:
                print("GraphConvLayerInitWeights: ", str(int(counter / self.graph_data.num_graphs * 100)), "%")
            cycle_dat = self.cycle_list[counter]
            graph_weight_pos_distribution = np.zeros((1, 3), np.dtype(np.int16))
            input_size = graph.number_of_nodes() * in_features
            weight_entry_num = 0
            W = torch.full((input_size, input_size * self.n_kernels), -1)
            for i in range(0, input_size):
                for j in range(0, input_size * self.n_kernels):
                    number = self.w_distribution_rule(i, j, self.in_features, self.n_node_labels, self.max_cycle_length,
                                                      self.n_kernels, graph, cycle_dat)
                    if number >= 0 and number < len(self.Param_W):
                        if weight_entry_num == 0:
                            graph_weight_pos_distribution[weight_entry_num][0] = i
                            graph_weight_pos_distribution[weight_entry_num][1] = j
                            graph_weight_pos_distribution[weight_entry_num][2] = number
                        else:
                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [[i, j, number]],
                                                                      axis=0)
                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            # print(graph_weight_pos_distribution)

        # Set the bias distribution for each graph
        self.bias_distribution = []
        for counter, graph in enumerate(self.graph_data.graphs, 0):
            if self.graph_data.num_graphs < 10 or counter % (self.graph_data.num_graphs // 10) == 0:
                print("GraphConvLayerInitBias: ", str(int(counter / self.graph_data.num_graphs * 100)), "%")
            graph_bias_pos_distribution = np.zeros((1, 2), np.dtype(np.int16))
            input_size = graph.number_of_nodes() * in_features
            bias_entry_num = 0
            bias = torch.full((input_size * self.n_kernels,), -1)
            # set bias weights
            bias_size = bias.size()[0]
            for i in range(0, bias_size):
                number = self.bias_distribution_rule(i, self.in_features, self.n_node_labels, self.n_kernels, graph)
                if number >= 0 and number < len(self.Param_b):
                    if bias_entry_num == 0:
                        graph_bias_pos_distribution[bias_entry_num][0] = i
                        graph_bias_pos_distribution[bias_entry_num][1] = number
                    else:
                        graph_bias_pos_distribution = np.append(graph_bias_pos_distribution, [[i, number]], axis=0)
                    bias_entry_num += 1

            self.bias_distribution.append(graph_bias_pos_distribution)
            # print(graph_bias_pos_distribution)

        self.forward_step_time = 0

    # set weights
    def set_weights(self, input_size, pos):
        W = torch.zeros((input_size, input_size * self.n_kernels))
        weight_distr = self.weight_distribution[pos]
        for entry in weight_distr:
            W[entry[0], entry[1]] = self.Param_W[entry[2]]
        return W

    # set bias weights
    def set_bias_weights(self, input_size, pos):
        bias = torch.zeros((input_size * self.n_kernels))
        bias_distr = self.bias_distribution[pos]
        for entry in bias_distr:
            bias[entry[0]] = self.Param_b[entry[1]]
        return bias

    def print_weights(self):
        print("Weights of the Convolution layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Convolution layer")
        for x in self.Param_b:
            print("\t", x.data)

    def forward(self, x, pos):
        x = x.view(-1).to("cpu")
        begin = time.time()
        """
        p = Process(target=set_weights())
        p.start()
        p.join()
        """
        W = self.set_weights(x.size()[0], pos).to("cpu")
        bias = self.set_bias_weights(x.size()[0], pos).to("cpu")

        self.forward_step_time += time.time() - begin

        # print(W)
        if self.bias:
            return torch.mv(W, x) + bias
        else:
            return torch.mv(W, x)


class GraphResizeLayer(nn.Module):
    def __init__(self, layer_id, seed, graph_data: GraphData.GraphData, w_distribution_rule, in_features, out_features,
                 node_labels, n_kernels=1,
                 bias=True, print_layer_init=False, save_weights=False, *args, **kwargs):
        super(GraphResizeLayer, self).__init__()

        self.layer_id = layer_id
        self.graph_data = graph_data

        self.in_features = in_features
        self.out_features = out_features

        self.node_labels = graph_data.node_labels[node_labels]
        n_node_labels = self.node_labels.num_unique_node_labels

        self.n_node_labels = n_node_labels
        self.weight_number = n_node_labels * out_features * in_features
        self.w_distribution_rule = w_distribution_rule
        self.n_kernels = n_kernels

        self.weight_num = in_features * n_kernels * n_node_labels * out_features
        self.weight_map = np.arange(self.weight_num, dtype=np.int16).reshape(
            (out_features, in_features, n_kernels, n_node_labels))

        self.weight_matrices = []

        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.Param_W = nn.ParameterList(
            [nn.Parameter(lower + torch.randn(1, dtype=torch.double) * (upper - lower)) for i in
             range(0, self.weight_number)])

        self.current_W = torch.Tensor()

        self.bias = bias
        if self.bias:
            self.Param_b = nn.Parameter(lower + torch.randn((1, out_features), dtype=torch.double) * (upper - lower))

        self.forward_step_time = 0

        self.name = "Resize_Layer"

        def valid_node_label(n_label):
            if 0 <= n_label < self.n_node_labels:
                return True
            else:
                return False

        # Set the distribution for each graph
        self.weight_distribution = []
        for graph_id, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or graph_id % (
                    self.graph_data.num_graphs // 10) == 0) and print_layer_init:
                print("ResizeLayerInitWeights: ", str(int(graph_id / self.graph_data.num_graphs * 100)), "%")

            node_labels = []
            graph_weight_pos_distribution = np.zeros((1, 3), np.dtype(np.int16))
            input_size = graph.number_of_nodes() * in_features * self.n_kernels
            weight_entry_num = 0
            in_high_dim = np.zeros(
                shape=(out_features, in_features, n_kernels, graph.number_of_nodes()))
            out_low_dim = np.zeros(shape=(out_features, input_size))
            index_map = reshape_indices(in_high_dim, out_low_dim)

            for o in range(0, out_features):
                for i1 in range(0, in_features):
                    for k in range(0, n_kernels):
                        for n1 in range(0, graph.number_of_nodes()):
                            n1_label = self.w_distribution_rule(n1, self.node_labels, graph_id)
                            if valid_node_label(int(n1_label)):
                                weight_pos = self.weight_map[o][i1][k][int(n1_label)]
                                if weight_entry_num == 0:
                                    graph_weight_pos_distribution[weight_entry_num][0] = index_map[(o, i1, k, n1)][0]
                                    graph_weight_pos_distribution[weight_entry_num][1] = index_map[(o, i1, k, n1)][1]
                                    graph_weight_pos_distribution[weight_entry_num][2] = weight_pos
                                else:
                                    graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [
                                        [index_map[(o, i1, k, n1)][0], index_map[(o, i1, k, n1)][1],
                                         weight_pos]], axis=0)
                                node_labels.append(int(n1_label))
                                weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)

            if save_weights:
                self.weight_matrices.append(torch.zeros((self.out_features, input_size), dtype=torch.double))
                parameterMatrix = np.full((self.out_features, input_size), 0, dtype=np.int16)
                for i, entry in enumerate(graph_weight_pos_distribution, 0):
                    self.weight_matrices[-1][entry[0]][entry[1]] = self.Param_W[entry[2]]
                    parameterMatrix[entry[0]][entry[1]] = entry[2] + 1
                    # save the parameter matrix
                    np.savetxt(
                        f"Results/{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt",
                        parameterMatrix, delimiter=';', fmt='%i')
            """
            for i in range(0, input_size):
                for j in range(0, input_size * self.n_kernels):
                    number = self.w_resize_distribution_rule(i, j, self.out_features, input_size, self.n_node_labels,
                                                             graph)
                    if number >= 0 and number < len(self.Param_W):
                        if weight_entry_num == 0:
                            graph_weight_pos_distribution[weight_entry_num][0] = i
                            graph_weight_pos_distribution[weight_entry_num][1] = j
                            graph_weight_pos_distribution[weight_entry_num][2] = number
                        else:
                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [[i, j, number]],
                                                                      axis=0)
                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            # print(graph_weight_pos_distribution)
            """

    def set_weights(self, input_size, pos):
        self.current_W = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=torch.double)
        num_graphs_nodes = self.graph_data.graphs[pos].number_of_nodes()
        weight_distr = self.weight_distribution[pos]
        for entry in weight_distr:
            self.current_W[entry[0]][entry[1]] = self.Param_W[entry[2]] / num_graphs_nodes

        # return self.weight_matrices[pos]

    def print_weights(self):
        print("Weights of the Resize layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Resize layer")
        for x in self.Param_b:
            print("\t", x.data)

    def forward(self, x, pos):
        x = x.view(-1)
        begin = time.time()
        self.set_weights(x.size()[0], pos)

        self.forward_step_time += time.time() - begin

        if self.bias:
            return torch.mv(self.current_W, x) + self.Param_b
        else:
            return torch.mv(self.current_W, x)

        # if self.bias:
        #     return torch.mv(self.weight_matrices[pos], x) + self.Param_b.to("cpu")
        # else:
        #     return torch.mv(self.weight_matrices[pos], x)

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b[0]]


# TODO: Modify this Layer such that it coincides with the NoGKernelNN using 1's as Input and as weights
class GraphResizeNoGLayer(nn.Module):
    def __init__(self, layer_id, seed, graph_data: GraphData.GraphData, w_distribution_rule, out_features,
                 n_node_labels, n_edge_labels=1,
                 bias=True, print_layer_init=False, save_weights=False, *args, **kwargs):
        super(GraphResizeLayer, self).__init__()

        self.graph_data = graph_data

        self.out_features = out_features
        self.n_node_labels = n_node_labels
        self.n_edge_labels = n_edge_labels
        self.weight_number = n_node_labels * n_edge_labels * out_features
        self.w_distribution_rule = w_distribution_rule

        self.weight_num = n_node_labels * n_edge_labels * out_features
        self.weight_map = np.arange(self.weight_num, dtype=np.int16).reshape(
            (out_features, n_node_labels, n_edge_labels))

        self.weight_matrices = []

        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.Param_W = nn.ParameterList(
            [nn.Parameter(lower + torch.randn(1, dtype=torch.double) * (upper - lower)) for i in
             range(0, self.weight_number)])

        self.current_W = torch.Tensor()

        self.bias = bias
        if self.bias:
            self.Param_b = nn.Parameter(lower + torch.randn((1, out_features), dtype=torch.double) * (upper - lower))

        self.forward_step_time = 0

        self.name = "Resize_Layer_NoG"

        def valid_node_label(n_label):
            if 0 <= n_label < self.n_node_labels:
                return True
            else:
                return False

        # Set the distribution for each graph
        self.weight_distribution = []
        for counter, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or counter % (
                    self.graph_data.num_graphs // 10) == 0) and print_layer_init:
                print("ResizeLayerInitWeights: ", str(int(counter / self.graph_data.num_graphs * 100)), "%")

            label_hist = gf.get_graph_label_hist(graph)
            node_labels = []
            graph_weight_pos_distribution = np.zeros((1, 3), np.dtype(np.int16))
            input_size = graph.number_of_nodes()
            weight_entry_num = 0
            in_high_dim = np.zeros(
                shape=(out_features, graph.number_of_nodes()))
            out_low_dim = np.zeros(shape=(out_features, input_size))
            index_map = reshape_indices(in_high_dim, out_low_dim)

            for o in range(0, out_features):
                for n1 in range(0, graph.number_of_nodes()):
                    n1_label = self.w_distribution_rule(n1, graph)
                    if valid_node_label(int(n1_label)):
                        weight_pos = self.weight_map[o][int(n1_label)]
                        if weight_entry_num == 0:
                            graph_weight_pos_distribution[weight_entry_num][0] = index_map[(o, n1)][0]
                            graph_weight_pos_distribution[weight_entry_num][1] = index_map[(o, n1)][1]
                            graph_weight_pos_distribution[weight_entry_num][2] = weight_pos
                        else:
                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [
                                [index_map[(o, n1)][0], index_map[(o, n1)][1],
                                 weight_pos]], axis=0)
                        node_labels.append(int(n1_label))
                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            self.weight_matrices.append(torch.zeros((self.out_features, input_size), dtype=torch.double))
            parameterMatrix = np.full((self.out_features, input_size), 0, dtype=np.int16)
            for i, entry in enumerate(graph_weight_pos_distribution, 0):
                self.weight_matrices[-1][entry[0]][entry[1]] = self.Param_W[entry[2]]
                parameterMatrix[entry[0]][entry[1]] = entry[2] + 1
            if save_weights:
                # save the parameter matrix
                np.savetxt(
                    f"Results/{graph_data.graph_db_name}/Weights/graph_{counter}_layer_{layer_id}_parameterWeightMatrix.txt",
                    parameterMatrix, delimiter=';', fmt='%i')
            """
            for i in range(0, input_size):
                for j in range(0, input_size * self.n_kernels):
                    number = self.w_resize_distribution_rule(i, j, self.out_features, input_size, self.n_node_labels,
                                                             graph)
                    if number >= 0 and number < len(self.Param_W):
                        if weight_entry_num == 0:
                            graph_weight_pos_distribution[weight_entry_num][0] = i
                            graph_weight_pos_distribution[weight_entry_num][1] = j
                            graph_weight_pos_distribution[weight_entry_num][2] = number
                        else:
                            graph_weight_pos_distribution = np.append(graph_weight_pos_distribution, [[i, j, number]],
                                                                      axis=0)
                        weight_entry_num += 1

            self.weight_distribution.append(graph_weight_pos_distribution)
            # print(graph_weight_pos_distribution)
            """

    def set_weights(self, input_size, pos):
        self.current_W = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=torch.double)
        # num_graphs_nodes = self.graph_data.graphs[pos].number_of_nodes()
        weight_distr = self.weight_distribution[pos]
        for entry in weight_distr:
            self.current_W[entry[0]][entry[1]] = self.Param_W[entry[2]]

        # return self.weight_matrices[pos]

    def print_weights(self):
        print("Weights of the Resize layer")
        for x in self.Param_W:
            print("\t", x.data)

    def print_bias(self):
        print("Bias of the Resize layer")
        for x in self.Param_b:
            print("\t", x.data)

    def forward(self, x, pos):
        x = x.view(-1)
        begin = time.time()
        self.set_weights(x.size()[0], pos)

        self.forward_step_time += time.time() - begin

        if self.bias:
            return torch.mv(self.current_W, x) + self.Param_b
        else:
            return torch.mv(self.current_W, x)

        # if self.bias:
        #     return torch.mv(self.weight_matrices[pos], x) + self.Param_b.to("cpu")
        # else:
        #     return torch.mv(self.weight_matrices[pos], x)
