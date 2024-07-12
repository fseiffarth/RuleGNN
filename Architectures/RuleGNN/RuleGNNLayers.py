'''
Created on 15.03.2019

@author:
'''
import torch
import torch.nn as nn
import torch.nn.init
import time
import numpy as np
from utils import GraphData
from utils.utils import reshape_indices


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
        self.sub_labels = None
        self.node_labels = -1
        self.layer_dict = layer_dict
        self.layer_id = layer_id

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
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "trivial":
            l_string = "trivial"
        elif self.layer_type == "combined":
            l_string = "combined"
            if 'id' in self.layer_dict:
                id = self.layer_dict['id']
                l_string = f"{l_string}_{id}"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string  = f"{l_string}_{max_node_labels}"

        return l_string


class RuleConvolutionLayer(nn.Module):
    """
    classdocs for the GraphConvLayer: This class represents a convolutional layer for a RuleGNN
    """

    def __init__(self, layer_id, seed, parameters, layer_info: Layer, graph_data: GraphData.GraphData,
                 in_features, n_kernels=1, bias=True,
                 precision=torch.float, device='cpu', *args,
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
        :param n_kernels: the number of kernels used in the layer (at the moment only 1 is supported)
        :param bias: if bias is used in the layer
        :param print_layer_init: if the layer initialization should be printed
        :param save_weights: if the weights should be saved
        :param precision: the precision of the weights, can be torch.float or torch.double
        :param args: additional arguments
        :param kwargs: additional keyword arguments
        """
        super(RuleConvolutionLayer, self).__init__()
        # id and name of the layer
        self.layer_id = layer_id
        # layer information
        self.layer_info = layer_info
        self.name = "WL_Layer"
        # get the graph data
        self.graph_data = graph_data
        # get the input features, i.e. the dimension of the input vector
        self.in_features = in_features
        # set the node labels
        self.node_labels = graph_data.node_labels[layer_info.get_layer_string()]
        # get the number of considered node labels
        self.n_node_labels = self.node_labels.num_unique_node_labels
        # get the number of considered kernels
        self.n_kernels = n_kernels

        self.n_properties = 1
        self.property = None
        if 'properties' in layer_info.layer_dict:
            self.property = graph_data.properties[layer_info.layer_dict['properties']['name']]
            self.n_properties = self.property.num_properties[layer_id]


        self.para = parameters  # get the all the parameters of the experiment
        self.bias = bias  # use bias or not default is True
        self.device = device  # set the device

        # Initialize the current weight matrix and bias vector
        self.current_W = torch.Tensor()
        self.current_B = torch.Tensor()

        self.args = args
        self.kwargs = kwargs
        self.precision = precision

        # Determine the number of weights and biases
        # There are two cases assymetric and symmetric, assymetric is the default
        if 'symmetric' in self.para.configs and self.para.configs['symmetric']:  #TODO
            self.weight_num = self.in_features * self.in_features * self.n_kernels * (
                        (self.n_node_labels * (self.n_node_labels + 1)) // 2) * self.n_properties
            # np upper triangular matrix
            self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
                (self.in_features, self.in_features, self.n_kernels, self.n_node_labels, self.n_node_labels, self.n_properties))
        else:
            self.weight_num = self.in_features * self.in_features * self.n_kernels * self.n_node_labels * self.n_node_labels * self.n_properties
            self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
                (self.in_features, self.in_features, self.n_kernels, self.n_node_labels, self.n_node_labels, self.n_properties))

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
        self.mask = None
        if 'prune' in self.para.configs and self.para.configs['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size())

        # Set the distribution for each graph
        self.weight_distribution = []
        # Set the bias distribution for each graph
        self.bias_distribution = []

        for graph_id, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or graph_id % (
                    self.graph_data.num_graphs // 10) == 0) and self.para.print_layer_init:
                print("GraphConvLayerInitWeights: ", str(int(graph_id / self.graph_data.num_graphs * 100)), "%")

            node_number = graph.number_of_nodes()  # get the number of nodes in the graph
            graph_weight_pos_distribution = []  # initialize the weight distribution

            input_size = node_number * self.in_features  # size of the weight matrix

            in_high_dim = np.zeros(
                shape=(self.in_features, self.in_features, self.n_kernels, node_number, node_number))
            out_low_dim = np.zeros(shape=(input_size, input_size * self.n_kernels))
            index_map = reshape_indices(in_high_dim, out_low_dim)

            for i1 in range(0, self.in_features):  # iterate over the input features, not used at the moment
                for i2 in range(0, self.in_features):  # iterate over the input features, not used at the moment
                    for k in range(0, self.n_kernels):  # iterate over the kernels, not used at the moment
                        # iterate over valid properties
                        for p in self.property.valid_property_map[layer_id].keys():
                            if p in self.property.properties[graph_id]:
                                for (v, w) in self.property.properties[graph_id][p]:
                                    v_label = self.node_labels.node_labels[graph_id][v]
                                    w_label = self.node_labels.node_labels[graph_id][w]
                                    property_id = self.property.valid_property_map[layer_id][p]
                                    # position of the weight in the Parameter list
                                    weight_pos = self.weight_map[i1][i2][k][int(v_label)][int(w_label)][property_id]
                                    # position of the weight in the weight matrix
                                    row_index = index_map[(i1, i2, k, v, w)][0]
                                    col_index = index_map[(i1, i2, k, v, w)][1]
                                    # vstack the new weight position
                                    graph_weight_pos_distribution.append([row_index, col_index, weight_pos])

            self.weight_distribution.append(np.array(graph_weight_pos_distribution, dtype=np.int64))

            graph_bias_pos_distribution = []
            out_size = node_number * self.in_features * self.n_kernels

            in_high_dim = np.zeros(shape=(self.in_features, self.n_kernels, node_number))
            out_low_dim = np.zeros(shape=(out_size,))
            index_map = reshape_indices(in_high_dim, out_low_dim)
            for i1 in range(0, self.in_features):  # not used at the moment
                for k in range(0, self.n_kernels):  # not used at the moment
                    for v in range(0, node_number):
                        v_label = self.node_labels.node_labels[graph_id][v]
                        bias_index = index_map[(i1, k, v)][0]
                        weight_pos = self.bias_map[i1][k][int(v_label)]
                        graph_bias_pos_distribution.append([bias_index, weight_pos])
            self.bias_distribution.append(np.array(graph_bias_pos_distribution, dtype=np.int64))

        self.forward_step_time = 0

    def set_weights(self, input_size, pos):
        # reshape self.current_W to the size of the weight matrix and fill it with zeros
        self.current_W = torch.zeros((input_size, input_size * self.n_kernels), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[pos]
        if len(weight_distr) != 0:
            # get third column of the weight_distribution: the index of self.Param_W
            param_indices = torch.tensor(weight_distr[:, 2]).long().to(self.device)
            matrix_indices = torch.tensor(weight_distr[:, 0:2]).T.long().to(self.device)
            # set current_W by using the matrix_indices with the values of the Param_W at the indices of param_indices
            self.current_W[matrix_indices[0], matrix_indices[1]] = torch.take(self.Param_W, param_indices)

    def set_bias(self, input_size, pos):
        self.current_B = torch.zeros((input_size * self.n_kernels), dtype=self.precision).to(self.device)
        bias_distr = self.bias_distribution[pos]

        self.current_B[bias_distr[:, 0]] = torch.take(self.Param_b, torch.tensor(bias_distr[:, 1]).to(self.device))

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

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b]


class RuleAggregationLayer(nn.Module):
    def __init__(self, layer_id, seed, parameters, layer_info: Layer, graph_data: GraphData.GraphData, in_features,
                 out_features, n_kernels=1,
                 bias=True, precision=torch.float, device='cpu'):
        super(RuleAggregationLayer, self).__init__()

        # id of the layer
        self.layer_id = layer_id
        # all the layer parameters
        self.layer_info = layer_info
        # get the graph data
        self.graph_data = graph_data
        self.precision = precision
        self.in_features = in_features
        self.out_features = out_features
        # device
        self.device = device

        self.node_labels = graph_data.node_labels[layer_info.get_layer_string()]
        n_node_labels = self.node_labels.num_unique_node_labels

        self.n_node_labels = n_node_labels
        self.weight_number = n_node_labels * out_features * in_features
        self.n_kernels = n_kernels

        self.weight_num = in_features * n_kernels * n_node_labels * out_features
        self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape(
            (out_features, in_features, n_kernels, n_node_labels))

        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.Param_W = nn.Parameter(lower + torch.randn(self.weight_number, dtype=self.precision) * (upper - lower))
        self.current_W = torch.Tensor()

        self.bias = bias
        if self.bias:
            self.Param_b = nn.Parameter(lower + torch.randn((1, out_features), dtype=self.precision) * (upper - lower))

        self.forward_step_time = 0

        self.name = "Resize_Layer"
        self.para = parameters

        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.configs and self.para.configs['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size(), requires_grad=False)

        # Set the distribution for each graph
        self.weight_distribution = []
        self.weight_normalization = []

        for graph_id, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or graph_id % (
                    self.graph_data.num_graphs // 10) == 0) and self.para.print_layer_init:
                print("ResizeLayerInitWeights: ", str(int(graph_id / self.graph_data.num_graphs * 100)), "%")

            graph_weight_pos_distribution = []
            input_size = graph.number_of_nodes() * in_features * self.n_kernels
            in_high_dim = np.zeros(
                shape=(out_features, in_features, n_kernels, graph.number_of_nodes()))
            out_low_dim = np.zeros(shape=(out_features, input_size))
            index_map = reshape_indices(in_high_dim, out_low_dim)

            ##########################################
            #weight_count_map = {}
            #weight_normal = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=self.precision)
            ##########################################

            for o in range(0, out_features):
                for i1 in range(0, in_features):
                    for k in range(0, n_kernels):
                        for v in range(0, graph.number_of_nodes()):
                            v_label = self.node_labels.node_labels[graph_id][v]
                            index_x = index_map[(o, i1, k, v)][0]
                            index_y = index_map[(o, i1, k, v)][1]
                            weight_pos = self.weight_map[o][i1][k][int(v_label)]
                            graph_weight_pos_distribution.append([index_x, index_y, weight_pos])

                            ##########################################
                            #if weight_pos in weight_count_map:
                            #    weight_count_map[weight_pos] += 1
                            #else:
                            #    weight_count_map[weight_pos] = 1
                            #weight_normal[index_x, index_y] = weight_pos
                            ##########################################

            # normalize the weights by their count (this is new compared to the paper)
            ##########################################
            #for key in weight_count_map:
            #    weight_normal[weight_normal == key] = 1 / (weight_count_map[key] * len(weight_count_map))
            #self.weight_normalization.append(weight_normal)
            ##########################################

            self.weight_distribution.append(np.array(graph_weight_pos_distribution, dtype=np.int64))

    def set_weights(self, input_size, pos):
        self.current_W = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=self.precision).to(self.device)
        num_graphs_nodes = self.graph_data.graphs[pos].number_of_nodes()
        weight_distr = self.weight_distribution[pos]
        param_indices = torch.tensor(weight_distr[:, 2]).long().to(self.device)
        matrix_indices = torch.tensor(weight_distr[:, 0:2]).T.long().to(self.device)
        self.current_W[matrix_indices[0], matrix_indices[1]] = torch.take(self.Param_W, param_indices)
        # divide the weights by the number of nodes in the graph
        self.current_W = self.current_W / num_graphs_nodes
        # normalize the weights using the weight_normalization
        #self.current_W = self.current_W * self.weight_normalization[pos]
        pass

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
