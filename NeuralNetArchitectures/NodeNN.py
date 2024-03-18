import Layers.GraphLayers as layers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import RuleFunctions.Rules as rule
from Time.TimeClass import TimeClass


###Net for learning graph class
class GraphNet(nn.Module):
    def __init__(self, graph_data, n_node_features, n_node_labels, n_edge_labels, distance_list=[], cycle_list=[],
                 print_weights=False):
        super(GraphNet, self).__init__()
        self.graph_data = graph_data
        self.print_weights = print_weights
        self.l1 = layers.GraphConvLayer(graph_data=self.graph_data, w_distribution_rule=rule.weight_rule_graphs_edge,
                                        bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=True)
        self.l2 = layers.GraphConvLayer(graph_data=self.graph_data, w_distribution_rule=rule.weight_rule_graphs_edge,
                                        bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features,
                                        n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1,
                                        bias=True)
        # self.l3 = layers.GraphConvLayer(graph_data=self.graph_data, w_distribution_rule=rule.weight_rule_graphs_edge, bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features, n_node_labels=n_node_labels, n_edge_labels=n_edge_labels, n_kernels=1, bias=True)
        # self.l4 = layers.GraphConvDistanceLayer(graph_data=self.graph_data, distance_list=distance_list, max_distance = 1, w_distribution_rule=rule.weight_rule_distances, bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features, out_features=100  , n_node_labels=n_node_labels, n_kernels=1, bias=True)
        # self.l5 = layers.GraphConvDistanceLayer(graph_data=self.graph_data, distance_list=distance_list, max_distance = 6, w_distribution_rule=rule.weight_rule_distances, bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features, out_features=100, n_node_labels=n_node_labels, n_kernels=1, bias=True)

        # self.lcycle = layers.GraphCycleLayer(graph_data=self.graph_data, cycle_list=cycle_list, max_cycle_length = 7, w_distribution_rule=rule.weight_rule_cycles, bias_distribution_rule=rule.bias_rule_graphs, in_features=n_node_features, out_features=100, n_node_labels=n_node_labels, n_kernels=1, bias=True)

        self.lr = layers.GraphResizeLayer(graph_data=self.graph_data,
                                          w_distribution_rule=rule.w_resize_distribution_rule,
                                          in_features=n_node_features, out_features=20, n_node_labels=n_node_labels,
                                          bias=True)
        self.lfc1 = nn.Linear(20, 20, bias=False)
        self.lfc2 = nn.Linear(20, 20, bias=False)
        self.lfc3 = nn.Linear(20, 1, bias=False)
        self.dropout = nn.Dropout(0.2)
        self.af = nn.Tanh()
        self.out_af = nn.Sigmoid()
        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        self.timer.measure("forward_wlrule")
        # x = self.dropout(x)

        x = self.af(self.l1(x, pos))
        x = self.af(self.l2(x, pos))
        self.timer.measure("forward_wlrule")
        # x = self.af(self.l4(x, pos))
        # x = self.af(self.l5(x, pos))
        # print(x)
        # x = self.af(self.lcycle(x, pos))
        """
        if self.epoch == 10:
            gdtgl.draw_graph_node_labels(self.graph_data[0][pos], torch.reshape(x, (len(self.graph_data[0][pos]), -1)).detach().numpy())
        """
        # x = self.af(self.l4(x, pos))
        """
        if self.epoch == 10:
            gdtgl.draw_graph_node_labels(self.graph_data[0][pos], torch.reshape(x, (len(self.graph_data[0][pos]), -1)).detach().numpy())
        """
        # x = self.af(self.l3(x, pos))
        self.timer.measure("forward_resize")
        x = self.af(self.lr(x, pos))
        self.timer.measure("forward_resize")
        self.timer.measure("forward_fc")
        x = self.af(self.lfc1(x))
        x = self.af(self.lfc2(x))
        x = self.out_af(self.lfc3(x))
        self.timer.measure("forward_fc")
        return x

    def return_info(self):
        return type(self)
