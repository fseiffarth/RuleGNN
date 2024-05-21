import Layers.GraphLayers as layers
import torch
import torch.nn as nn
import RuleFunctions.Rules as rule
from GraphData import GraphData

from Time.TimeClass import TimeClass
from utils.Parameters.Parameters import Parameters


class GraphNet(nn.Module):
    def __init__(self, graph_data: GraphData, para: Parameters, seed):
        super(GraphNet, self).__init__()
        self.graph_data = graph_data
        self.para = para
        self.print_weights = self.para.net_print_weights
        self.module_precision = 'double'
        n_node_features = self.para.node_features
        dropout = self.para.dropout
        print_layer_init = self.para.print_layer_init
        save_weights = self.para.save_weights
        convolution_grad = self.para.convolution_grad
        resize_grad = self.para.resize_grad
        out_classes = self.graph_data.num_classes
        if 'precision' in para.configs:
            self.module_precision = para.configs['precision']

        self.net_layers = nn.ModuleList()
        for i, layer in enumerate(para.layers):
            if i < len(para.layers) - 1:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        layers.GraphConvLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                              w_distribution_rule=rule.weight_rule_wf_dist,
                                              bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                              node_labels_name=layer.get_layer_string(), n_kernels=1,
                                              bias=True, print_layer_init=print_layer_init, save_weights=save_weights,
                                              distances=layer.distances, precision=torch.float).float().requires_grad_(convolution_grad))
                else:
                    self.net_layers.append(
                        layers.GraphConvLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                              w_distribution_rule=rule.weight_rule_wf_dist,
                                              bias_distribution_rule=rule.node_label_rule, in_features=n_node_features,
                                              node_labels_name=layer.get_layer_string(), n_kernels=1,
                                              bias=True, print_layer_init=print_layer_init, save_weights=save_weights,
                                              distances=layer.distances, precision=torch.double).double().requires_grad_(convolution_grad))
            else:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        layers.GraphResizeLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                                w_distribution_rule=rule.node_label_rule,
                                                in_features=n_node_features, out_features=out_classes,
                                                node_labels=layer.get_layer_string(),
                                                bias=True, print_layer_init=print_layer_init,
                                                save_weights=save_weights, precision=torch.float).float().requires_grad_(
                            resize_grad))
                else:
                    self.net_layers.append(
                        layers.GraphResizeLayer(layer_id=i, seed=seed + i, parameters=para, graph_data=self.graph_data,
                                                w_distribution_rule=rule.node_label_rule,
                                                in_features=n_node_features, out_features=out_classes,
                                                node_labels=layer.get_layer_string(),
                                                bias=True, print_layer_init=print_layer_init,
                                                save_weights=save_weights, precision=torch.double).double().requires_grad_(
                            resize_grad))

        if 'linear_layers' in para.configs and para.configs['linear_layers'] > 0:
            for i in range(para.configs['linear_layers']):
                if self.module_precision == 'float':
                    self.net_layers.append(nn.Linear(out_classes, out_classes, bias=True).float())
                else:
                    self.net_layers.append(nn.Linear(out_classes, out_classes, bias=True).double())

        self.dropout = nn.Dropout(dropout)
        if 'activation' in para.configs and para.configs['activation'] == 'None':
            self.af = nn.Identity()
        elif 'activation' in para.configs and para.configs['activation'] == 'Relu':
            self.af = nn.ReLU()
        elif 'activation' in para.configs and para.configs['activation'] == 'LeakyRelu':
            self.af = nn.LeakyReLU()
        else:
            self.af = nn.Tanh()
        if 'output_activation' in para.configs and para.configs['output_activation'] == 'None':
            self.out_af = nn.Identity()
        elif 'output_activation' in para.configs and para.configs['output_activation'] == 'Relu':
            self.out_af = nn.ReLU()
        elif 'output_activation' in para.configs and para.configs['output_activation'] == 'LeakyRelu':
            self.out_af = nn.LeakyReLU()
        else:
            self.out_af = nn.Tanh()
        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        for i, layer in enumerate(self.net_layers):
            if i < len(self.net_layers) - 1:
                x = self.af(layer(x, pos))
            else:
                if 'linear_layers' in self.para.configs and self.para.configs['linear_layers'] > 0:
                    x = self.out_af(layer(x))
                else:
                    x = self.out_af(layer(x, pos))
        return x

    def return_info(self):
        return type(self)