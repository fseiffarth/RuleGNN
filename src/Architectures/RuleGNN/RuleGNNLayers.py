'''
Created on 15.03.2019

@author:
'''
from typing import Tuple

import matplotlib
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.init
import time
import numpy as np
from sympy.physics.units import second

from src.utils import GraphData, GraphDrawing


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
        elif self.layer_type == "index":
            l_string = "index"
            if 'max_node_labels' in self.layer_dict:
                max_node_labels = self.layer_dict['max_node_labels']
                l_string = f"index_{max_node_labels}"
        elif self.layer_type == "wl":
            iterations = self.layer_dict.get('wl_iterations', 3)
            l_string = f"wl_{iterations}"
            max_node_labels = self.layer_dict.get('max_node_labels', None)
            if max_node_labels is not None:
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "wl_labeled":
            iterations = self.layer_dict.get('wl_iterations', 3)
            l_string = f"wl_labeled_{iterations}"
            max_node_labels = self.layer_dict.get('max_node_labels', None)
            if max_node_labels is not None:
                l_string = f"{l_string}_{max_node_labels}"
        elif self.layer_type == "degree":
            l_string = "wl_0"
            max_node_labels = self.layer_dict.get('max_node_labels', None)
            if max_node_labels is not None:
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
        else:
            raise ValueError(f"Layer type {self.layer_type} is not supported")

        return l_string

class RuleConvolutionLayer(nn.Module):
    """
    classdocs for the GraphConvLayer: This class represents a convolutional layer for a RuleGNN
    """

    def __init__(self, layer_id, seed, parameters, layer_info: Layer, graph_data: GraphData.GraphData, bias=True, out_channels=1, device='cpu'):
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
        self.layer_info = layer_info
        self.name = f"Rule Convolution Layer: {layer_info.get_layer_string()}"
        # get the graph data
        self.graph_data = graph_data
        # get the input features, i.e. the dimension of the input vector
        self.input_feature_dimension = graph_data.input_feature_dimension
        # set the node labels
        self.node_labels = graph_data.node_labels[layer_info.get_layer_string()]
        # get the number of considered node labels
        self.n_node_labels = self.node_labels.num_unique_node_labels
        # get the number of input channels
        self.out_channels = out_channels


        self.n_properties = 1
        self.property = None
        if 'properties' in layer_info.layer_dict:
            self.property = graph_data.properties[layer_info.layer_dict['properties']['name']]
            self.n_properties = self.property.num_properties[layer_id]


        self.para = parameters  # get the all the parameters of the experiment
        self.bias = bias  # use bias or not default is True
        self.device = device  # set the device
        self.precision = torch.float # set the precision of the weights
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double

        # Initialize the current weight matrix, bias vector and feature weight matrix
        self.current_W = torch.Tensor()
        self.current_B = torch.Tensor()
        self.feature_W = torch.Tensor()
        if parameters.run_config.config.get('use_feature_weights', False):
            self.feature_W = nn.Parameter(torch.randn((self.input_feature_dimension, self.input_feature_dimension), dtype=self.precision))


        # Determine the number of weights and biases
        # There are two cases assymetric and symmetric, assymetric is the default
        if self.para.run_config.config.get('symmetric', False):
            self.weight_num = self.out_channels * ((self.n_node_labels * (self.n_node_labels + 1)) // 2) * self.n_properties
        else:
            self.weight_num = self.out_channels * self.n_node_labels * self.n_node_labels * self.n_properties
        if self.bias:
            # Determine the number of different learnable parameters in the bias vector
            self.bias_num = self.input_feature_dimension * self.out_channels * self.n_node_labels
            self.bias_map = np.arange(self.bias_num, dtype=np.int64).reshape((self.out_channels, self.n_node_labels, self.input_feature_dimension))

        if self.bias:
            self.Param_b = nn.Parameter(torch.zeros(self.bias_num, dtype=self.precision), requires_grad=True)

        # Set the distribution for each graph
        self.weight_distribution = []
        # Set the bias distribution for each graph
        self.bias_distribution = []
        # list of degree matrices
        self.D = []



        for graph_id, graph in enumerate(self.graph_data.graphs, 0):
            if (self.graph_data.num_graphs < 10 or graph_id % (
                    self.graph_data.num_graphs // 10) == 0) and self.para.print_layer_init:
                print("GraphConvLayerInitWeights: ", str(int(graph_id / self.graph_data.num_graphs * 100)), "%")

            node_number = graph.number_of_nodes()  # get the number of nodes in the graph
            graph_weight_pos_distribution = [] # initialize the weight distribution # size of the weight matrix


            if self.para.run_config.config.get('degree_matrix', False):
                self.D.append(torch.zeros(node_number, dtype=self.precision).to(self.device))


            for k in range(0, self.out_channels):  # iterate over the channels, not used at the moment
                # iterate over valid properties
                for p in self.property.valid_property_map[layer_id].keys():
                    if p in self.property.properties[graph_id]:
                        for (v, w) in self.property.properties[graph_id][p]:
                            v_label = self.node_labels.node_labels[graph_id][v]
                            w_label = self.node_labels.node_labels[graph_id][w]
                            property_id = self.property.valid_property_map[layer_id][p]
                            # position of the weight in the Parameter list
                            #weight_pos = self.weight_map[k][int(v_label)][int(w_label)][property_id]
                            if self.para.run_config.config.get('symmetric', False):
                                weight_pos = self.symmetric_weight_map(k, int(v_label), int(w_label), property_id)
                            else:
                                weight_pos = self.asymmetric_weight_map(k, int(v_label), int(w_label), property_id)
                            # position of the weight in the weight matrix
                            #row_index = index_map[(k, v, w)][0]
                            #col_index = index_map[(k, v, w)][1]
                            # vstack the new weight position
                            graph_weight_pos_distribution.append([k, v, w, weight_pos])
                            # add entry to the degree matrix
                            if self.para.run_config.config.get('degree_matrix', False):
                                self.D[graph_id][v] += 1.0
                                self.D[graph_id][w] += 1.0
            self.weight_distribution.append(np.array(graph_weight_pos_distribution, dtype=np.int64))

            # normalize the degree matrix to inverse square root
            if self.para.run_config.config.get('degree_matrix', False):
                self.D[graph_id] = torch.pow(self.D[graph_id], -0.5)

            if self.bias:
                graph_bias_pos_distribution = []
                for k in range(0, self.out_channels):
                    for i in range(0, self.input_feature_dimension):  # not used at the moment # not used at the moment
                        for v in range(0, node_number):
                            v_label = self.node_labels.node_labels[graph_id][v]
                            weight_pos = self.bias_map[k][int(v_label)][i]
                            graph_bias_pos_distribution.append([k, v, i, weight_pos])

                self.bias_distribution.append(np.array(graph_bias_pos_distribution, dtype=np.int64))

        # consider only the rules that appear in the dataset
        weight_occurences = np.zeros(self.weight_num)
        for i, weights in enumerate(self.weight_distribution):
            weight_pos = weights[:, 3]
            # in weight_array add 1 where the index is in weight_pos
            weight_occurences[weight_pos] += 1
        # get all the non-zero weights
        non_zero_positions = np.nonzero(weight_occurences)
        num_non_zero_weights = non_zero_positions[0].shape[0]
        # map from non-zero weights to indices
        non_zero_weight_map = {weight_idx: pos_idx for pos_idx, weight_idx in enumerate(non_zero_positions[0])}
        self.Param_W = nn.Parameter(torch.zeros(num_non_zero_weights, dtype=self.precision), requires_grad=True)
        # initialize self.Param_W with small non-zero values
        torch.nn.init.constant_(self.Param_W, 0.001)
        # modify the weight distribution to only contain the non-zero weights
        for i, weights in enumerate(self.weight_distribution):
            weight_pos = weights[:, 3]
            new_weight_pos = np.array([non_zero_weight_map[pos] for pos in weight_pos])
            # in weight_array add 1 where the index is in weight_pos
            self.weight_distribution[i][:, 3] = new_weight_pos

        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
            self.Param_W_original = self.Param_W.detach().clone()
            self.mask = torch.ones(self.Param_W.size())

        self.forward_step_time = 0

    def asymmetric_weight_map(self, channel, label1, label2, property_id)-> int:
        '''
        Maps the current indices to the index of the list of weights
        '''
        first_skip_size = self.n_node_labels * self.n_node_labels * self.n_properties
        second_skip_size = self.n_node_labels * self.n_properties
        third_skip_size = self.n_properties
        return channel * first_skip_size + label1 * second_skip_size + label2 * third_skip_size + property_id

    def symmetric_weight_map(self, channel, label1, label2, property_id)-> int:
        '''
        Maps the current indices to the index of the list of weights
        '''
        first_skip_size = (self.n_node_labels * (self.n_node_labels + 1)) // 2 * self.n_properties
        second_step_size = self.n_properties
        min_label = min(label1, label2)
        max_label = max(label1, label2)
        pos_from_labels = min_label * (2 * self.n_node_labels - min_label + 1) // 2 + (max_label - min_label)
        return channel * first_skip_size + pos_from_labels * second_step_size + property_id

    def set_weights(self, pos):
        input_size = self.graph_data.graphs[pos].number_of_nodes()
        # reshape self.current_W to the size of the weight matrix and fill it with zeros
        self.current_W = torch.zeros((self.out_channels, input_size, input_size), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[pos]
        if len(weight_distr) != 0:
            # get third column of the weight_distribution: the index of self.Param_W
            param_indices = torch.tensor(weight_distr[:, 3]).long().to(self.device)
            matrix_indices = torch.tensor(weight_distr[:, 0:3]).T.long().to(self.device)
            # set current_W by using the matrix_indices with the values of the Param_W at the indices of param_indices
            self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, param_indices)

    def set_bias(self, pos):
        input_size = self.graph_data.graphs[pos].number_of_nodes()
        self.current_B = torch.zeros((self.out_channels, input_size, self.input_feature_dimension), dtype=self.precision).to(self.device)
        bias_distr = self.bias_distribution[pos]
        param_indices = torch.tensor(bias_distr[:, 3]).long().to(self.device)
        matrix_indices = torch.tensor(bias_distr[:, 0:3]).T.long().to(self.device)
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
                return torch.einsum('cij,cjk->cik', torch.diag(self.D[pos]) @ self.current_W @ torch.diag(self.D[pos]), x) + self.current_B
            else:
                return torch.einsum('cij,cjk->cik', self.current_W, x) + self.current_B
        else:
            self.forward_step_time += time.time() - begin
            if self.para.run_config.config.get('degree_matrix', False):
                return torch.einsum('cij,cjk->cik', torch.diag(self.D[pos]) @ self.current_W @ torch.diag(self.D[pos]), x)
            else:
                return torch.einsum('cij,cjk->cik', self.current_W, x)


    def get_weights(self):
        # return the weights as a numpy array
        return np.array(self.Param_W.detach().cpu())

    def get_bias(self):
        return np.array(self.Param_b.detach().cpu())

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], channel=0, filter_weights=None, with_graph=True, graph_only=False):
        if with_graph or graph_only:
            graph = self.graph_data.graphs[graph_id]

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

        # sort weights
        if filter_weights is not None:
            sorted_weights = np.sort(graph_weights)
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
        for node in digraph.nodes():
            node_label = self.node_labels.node_labels[graph_id][node]
            node_colors.append(bias_colors[node_label])
            node_sizes.append(graph_drawing[1].node_size * abs(bias[node_label]) / bias_max_abs)

        nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)

        edge_widths = []
        for weight_id, entry in enumerate(weight_distribution):
            c = entry[0]
            if c == channel:
                i = entry[1]
                j = entry[2]
                if weights[weight_id] != 0:
                    # add edge with weight as data
                    digraph.add_edge(i, j, weight=weight_id)
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
    def __init__(self, layer_id, seed, parameters, layer_info: Layer, graph_data: GraphData.GraphData,
                 out_dim, out_channels=1, bias=True, device='cpu'):

        super(RuleAggregationLayer, self).__init__()

        # id of the layer
        self.layer_id = layer_id
        # all the layer parameters
        self.layer_info = layer_info
        # get the graph data
        self.graph_data = graph_data
        self.precision = torch.float
        if parameters.run_config.config.get('precision', 'float') == 'double':
            self.precision = torch.double
        self.channels = out_channels
        self.output_dimension = out_dim
        # device
        self.device = device

        self.node_labels = graph_data.node_labels[layer_info.get_layer_string()]
        n_node_labels = self.node_labels.num_unique_node_labels

        self.n_node_labels = n_node_labels
        self.input_feature_dimension = graph_data.input_feature_dimension

        self.weight_num = n_node_labels * out_dim * self.channels
        self.weight_map = np.arange(self.weight_num, dtype=np.int64).reshape((self.channels, out_dim, n_node_labels))

        # calculate the range for the weights
        lower, upper = -(1.0 / np.sqrt(self.weight_num)), (1.0 / np.sqrt(self.weight_num))
        # set seed for reproducibility
        torch.manual_seed(seed)
        self.Param_W = nn.Parameter(lower + torch.randn(self.weight_num, dtype=self.precision) * (upper - lower))
        self.current_W = torch.Tensor()
        torch.nn.init.constant_(self.Param_W, 0.0001)

        self.bias = bias
        if self.bias:
            self.Param_b = nn.Parameter(torch.zeros((self.channels, out_dim, self.input_feature_dimension), dtype=self.precision))
        self.forward_step_time = 0

        self.name = f"Rule Aggregation Layer: {layer_info.get_layer_string()}"
        self.para = parameters

        # in case of pruning is turned on, save the original weights
        self.Param_W_original = None
        self.mask = None
        if 'prune' in self.para.run_config.config and self.para.run_config.config['prune']['enabled']:
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

            ##########################################
            #weight_count_map = {}
            #weight_normal = torch.zeros((self.out_features, input_size * self.n_kernels), dtype=self.precision)
            ##########################################

            for c in range(0, self.channels):
                for o in range(0, out_dim):
                        for v in range(0, graph.number_of_nodes()):
                            v_label = self.node_labels.node_labels[graph_id][v]
                            weight_pos = self.weight_map[c][o][int(v_label)]
                            graph_weight_pos_distribution.append([c, o, v, weight_pos])

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

    def set_weights(self, pos):
        input_size = self.graph_data.graphs[pos].number_of_nodes()
        self.current_W = torch.zeros((self.channels, self.output_dimension, input_size), dtype=self.precision).to(self.device)
        weight_distr = self.weight_distribution[pos]
        param_indices = torch.tensor(weight_distr[:, 3]).long().to(self.device)
        matrix_indices = torch.tensor(weight_distr[:, 0:3]).T.long().to(self.device)
        self.current_W[matrix_indices[0], matrix_indices[1], matrix_indices[2]] = torch.take(self.Param_W, param_indices)
        # divide the weights by the number of nodes in the graph
        self.current_W = self.current_W / input_size
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
            return torch.einsum('cij,cjk->cik', self.current_W, x) + self.Param_b
        else:
            return torch.einsum('cij,cjk->cik', self.current_W, x)

        # if self.bias:
        #     return torch.mv(self.weight_matrices[pos], x) + self.Param_b.to("cpu")
        # else:
        #     return torch.mv(self.weight_matrices[pos], x)

    def get_weights(self):
        return [x.item() for x in self.Param_W]

    def get_bias(self):
        return [x.item() for x in self.Param_b[0]]

    def draw(self, ax, graph_id, graph_drawing: Tuple[GraphDrawing, GraphDrawing], channel=0, out_dimension=0, with_graph=True, graph_only=False):
        if with_graph or graph_only:
            graph = self.graph_data.graphs[graph_id]

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
            if c == channel and o_dimension == out_dimension:
                node_colors.append(weight_colors[i])
                node_sizes.append(graph_drawing[1].node_size * abs(graph_weights[i]) / weight_max_abs)

        nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)