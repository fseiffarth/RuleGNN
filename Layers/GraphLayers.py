'''
Created on 15.03.2019

@author: florian
'''
import torch
import torch.nn as nn
import torch.nn.init
import time
import numpy as np
import ReadWriteGraphs.GraphFunctions as gf
from GraphData import GraphData
from utils.RunConfiguration import RunConfiguration


class Layer:
    """
    classdocs for the Layer: This class represents a layer in a RuleGNN
    """

    def __init__(self, layer_dict):
        """
        Constructor of the Layer
        :param layer_dict: the dictionary that contains the layer information
        """
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
        """
        Method to get the layer string. This is used to load the node labels
        """
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
            if 'id' in self.layer_dict:
                id = self.layer_dict['id']
                l_string = f"{l_string}_{id}"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"subgraph_{max_node_labels}"
        elif self.layer_type == "trivial":
            l_string = "trivial"

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
    """
    classdocs for the GraphConvLayer: This class represents a convolutional layer for a RuleGNN
    """

    def __init__(self, layer_id, seed, parameters, graph_data: GraphData.GraphData, w_distribution_rule,
                 bias_distribution_rule,
                 in_features, node_labels_name, n_kernels=1, bias=True, print_layer_init=False, save_weights=False,
                 precision=torch.float, *args,
                 **kwargs):
        """
        Constructor of the GraphConvLayer
        :param layer_id: the id of the layer
        :param seed: the seed for the random number generator
        :param parameters: the parameters of the experiment
        :param graph_data: the data of the graph dataset
        :param w_distribution_rule: the rule for the weight distribution in the layer
        :param bias_distribution_rule: the rule for the bias distribution in the layer
        :param in_features: the number of input features (at the moment only 1 is supported)
        :param node_labels_name: the name of the node labels used in the layer
        :param n_kernels: the number of kernels used in the layer (at the moment only 1 is supported)
        :param bias: if bias is used in the layer
        :param print_layer_init: if the layer initialization should be printed
        :param save_weights: if the weights should be saved
        :param precision: the precision of the weights, can be torch.float or torch.double
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        """
        super(GraphConvLayer, self).__init__()
        # id and name of the layer
        self.layer_id = layer_id
        self.name = "WL_Layer"
        # get the graph data
        self.graph_data = graph_data
        # get the input features, i.e. the dimension of the input vector
        self.in_features = in_features
        # set the node labels
        self.node_labels = graph_data.node_labels[node_labels_name]
        # get the number of considered node labels
        self.n_node_labels = self.node_labels.num_unique_node_labels
        self.edge_labels = graph_data.edge_labels['primary']
        # get the number of considered edge labels
        self.n_edge_labels = self.edge_labels.num_unique_edge_labels
        # get the number of considered kernels
        self.n_kernels = n_kernels
        self.n_extra_dim = 1
        self.extra_dim_map = {}
        self.para = parameters  # get the all the parameters of the experiment
        self.bias = bias  # use bias or not default is True

        # Initialize the current weight matrix and bias vector
        self.current_W = torch.Tensor()
        self.current_B = torch.Tensor()
        # Get the rules for the weight and bias distribution
        self.w_distribution_rule = w_distribution_rule
        self.bias_distribution_rule = bias_distribution_rule

        self.args = args
        self.kwargs = kwargs
        self.precision = precision

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
        # There are two cases assymetric and symmetric, assymetric is the default
        if 'symmetric' in self.para.configs and self.para.configs['symmetric']:  #TODO
            self.weight_num = self.in_features * self.in_features * self.n_kernels * (
                        (self.n_node_labels * (self.n_node_labels + 1)) // 2) * self.n_edge_labels * self.n_extra_dim
            # np upper triangular matrix
            self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
                (self.in_features, self.in_features, self.n_kernels, self.n_node_labels, self.n_node_labels,
                 self.n_edge_labels, self.n_extra_dim))
        else:
            self.weight_num = self.in_features * self.in_features * self.n_kernels * self.n_node_labels * self.n_node_labels * self.n_edge_labels * self.n_extra_dim
            self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
                (self.in_features, self.in_features, self.n_kernels, self.n_node_labels, self.n_node_labels,
                 self.n_edge_labels, self.n_extra_dim))

        # Determine the number of different learnable parameters in the bias vector
        self.bias_num = self.in_features * self.n_kernels * self.n_node_labels
        self.bias_map = np.arange(self.bias_num, dtype=np.int64).reshape(
            (self.in_features, self.n_kernels, self.n_node_labels))

        # calculate the range for the weights using the number of weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))

        # set seed for reproducibility
        torch.manual_seed(seed)
        # Initialize the weight matrix with random values between lower and upper
        weight_data = lower + torch.randn(self.weight_num, dtype=self.precision) * (upper - lower)
        self.Param_W = nn.Parameter(weight_data, requires_grad=True)
        bias_data = lower + torch.randn(self.bias_num, dtype=self.precision) * (upper - lower)
        self.Param_b = nn.Parameter(bias_data, requires_grad=True)

        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        if 'prune' in self.para.configs and self.para.configs['prune']:
            self.Param_W_original = self.Param_W.detach().clone()

        def valid_node_label(n_label: int) -> bool:
            """
            Check if the node label is valid, i.e., it is in the range of the node labels
            :param n_label: the node label
            :return: True if the node label is valid, False otherwise
            """
            if 0 <= n_label < self.n_node_labels:
                return True
            else:
                return False

        def valid_edge_label(edge_label: int, src: int = 0, dst: int = 1) -> bool:
            """
            Check if the edge label is valid, i.e., it is in the range of the edge labels
            :param edge_label: the edge label
            :param src: the source node
            :param dst: the destination node
            :return: True if the edge label is valid, False otherwise
            """
            if 0 <= edge_label < self.n_edge_labels:
                return True
            elif src == dst and 0 <= edge_label < self.n_edge_labels + 1:
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
            graph_weight_pos_distribution = np.zeros((1, 3), dtype=np.int64)

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
                self.weight_matrices.append(
                    torch.zeros((input_size, input_size * self.n_kernels), dtype=self.precision))
                for entry in graph_weight_pos_distribution:
                    self.weight_matrices[-1][entry[0]][entry[1]] = self.Param_W[entry[2]]
                    parameterMatrix[entry[0]][entry[1]] = entry[2] + 1
                    np.savetxt(
                        f"{self.para.configs['paths']['results']}/{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt",
                        parameterMatrix, delimiter=';', fmt='%i')

            # print(row_array, col_array, data_array)
            # self.sparse_weight_data.append(data_array)
            # self.sparse_weight_row_col.append(torch.cat((row_array, col_array), 0))
            # print(self.sparse_weight_data, self.sparse_weight_row_col)

            graph_bias_pos_distribution = np.zeros((1, 2), np.dtype(np.int16))
            out_size = node_number * self.in_features * self.n_kernels
            bias = torch.zeros((out_size), dtype=self.precision)

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
        # reshape self.current_W to the size of the weight matrix and fill it with zeros
        self.current_W = torch.zeros((input_size, input_size * self.n_kernels), dtype=self.precision)
        weight_distr = self.weight_distribution[pos]
        # get third column of the weight_distribution: the index of self.Param_W
        param_indices = torch.tensor(weight_distr[:, 2]).long()
        matrix_indices = torch.tensor(weight_distr[:, 0:2]).T.long()
        # set current_W by using the matrix_indices with the values of the Param_W at the indices of param_indices
        self.current_W[matrix_indices[0], matrix_indices[1]] = torch.take(self.Param_W, param_indices)
        #self.current_W = self.weight_matrices[pos]

    def set_bias(self, input_size, pos):
        self.current_B = torch.zeros((input_size * self.n_kernels), dtype=self.precision)
        bias_distr = self.bias_distribution[pos]

        self.current_B[bias_distr[:, 0]] = torch.take(self.Param_b, torch.tensor(bias_distr[:, 1]))

        #for entry in bias_distr:
        #self.current_B[entry[0]] = self.Param_b[entry[1]]

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


class GraphResizeLayer(nn.Module):
    def __init__(self, layer_id, seed, parameters, graph_data: GraphData.GraphData, w_distribution_rule, in_features,
                 out_features,
                 node_labels, n_kernels=1,
                 bias=True, print_layer_init=False, save_weights=False, precision=torch.float, *args, **kwargs):
        super(GraphResizeLayer, self).__init__()

        self.layer_id = layer_id
        self.graph_data = graph_data

        self.precision = precision

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
        self.Param_W = nn.Parameter(lower + torch.randn(self.weight_number, dtype=self.precision) * (upper - lower))

        #self.Param_W = nn.ParameterList(
        #    [nn.Parameter(lower + torch.randn(1, dtype=self.precision) * (upper - lower)) for i in
        #     range(0, self.weight_number)])

        self.current_W = torch.Tensor()

        self.bias = bias
        if self.bias:
            self.Param_b = nn.Parameter(lower + torch.randn((1, out_features), dtype=self.precision) * (upper - lower))

        self.forward_step_time = 0

        self.name = "Resize_Layer"
        self.para = parameters

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
                self.weight_matrices.append(torch.zeros((self.out_features, input_size), dtype=self.precision))
                parameterMatrix = np.full((self.out_features, input_size), 0, dtype=np.int16)
                for i, entry in enumerate(graph_weight_pos_distribution, 0):
                    self.weight_matrices[-1][entry[0]][entry[1]] = self.Param_W[entry[2]]
                    parameterMatrix[entry[0]][entry[1]] = entry[2] + 1
                    # save the parameter matrix
                    np.savetxt(
                        f"{self.para.configs['paths']['results']}/{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt",
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
        self.current_W = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=self.precision)
        num_graphs_nodes = self.graph_data.graphs[pos].number_of_nodes()
        weight_distr = self.weight_distribution[pos]
        param_indices = torch.tensor(weight_distr[:, 2])
        matrix_indices = torch.tensor(weight_distr[:, 0:2]).T
        self.current_W[matrix_indices[0], matrix_indices[1]] = torch.take(self.Param_W, param_indices)
        # divide the weights by the number of nodes in the graph
        self.current_W = self.current_W / num_graphs_nodes

        #for entry in weight_distr:
        #    self.current_W[entry[0]][entry[1]] = self.Param_W[entry[2]] / num_graphs_nodes

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
