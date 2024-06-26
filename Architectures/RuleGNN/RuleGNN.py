import Architectures.RuleGNN.RuleGNNLayers as layers
import torch
import torch.nn as nn
from utils import GraphData

from Time.TimeClass import TimeClass
from utils.Parameters.Parameters import Parameters


class RuleGNN(nn.Module):
    def __init__(self, graph_data: GraphData, para: Parameters, seed):
        super(RuleGNN, self).__init__()
        self.graph_data = graph_data
        self.para = para
        self.print_weights = self.para.net_print_weights
        self.module_precision = 'double'
        n_node_features = self.para.node_features
        dropout = self.para.dropout
        convolution_grad = self.para.convolution_grad
        resize_grad = self.para.resize_grad
        out_classes = self.graph_data.num_classes
        if 'precision' in para.configs:
            self.module_precision = para.configs['precision']
        if 'aggregation_out_features' in para.configs:
            self.aggregation_out_classes = para.configs['aggregation_out_features']
        else:
            self.aggregation_out_classes = out_classes


        self.net_layers = nn.ModuleList()
        for i, layer in enumerate(para.layers):
            if i < len(para.layers) - 1:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        layers.RuleConvolutionLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    in_features=n_node_features, n_kernels=1,
                                                    bias=True, precision=torch.float).float().requires_grad_(
                            convolution_grad))
                else:
                    self.net_layers.append(
                        layers.RuleConvolutionLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    in_features=n_node_features, n_kernels=1,
                                                    bias=True, precision=torch.double).double().requires_grad_(
                            convolution_grad))
            else:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        layers.RuleAggregationLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    in_features=n_node_features, out_features=self.aggregation_out_classes,
                                                    bias=True,
                                                    precision=torch.float).float().requires_grad_(
                            resize_grad))
                else:
                    self.net_layers.append(
                        layers.RuleAggregationLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    in_features=n_node_features, out_features=self.aggregation_out_classes,
                                                    bias=True,
                                                    precision=torch.double).double().requires_grad_(
                            resize_grad))

        if 'linear_layers' in para.configs and para.configs['linear_layers'] > 0:
            for i in range(para.configs['linear_layers']):
                if i < para.configs['linear_layers'] - 1:
                    if self.module_precision == 'float':
                        self.net_layers.append(nn.Linear(self.aggregation_out_classes, self.aggregation_out_classes, bias=True).float())
                    else:
                        self.net_layers.append(nn.Linear(self.aggregation_out_classes, self.aggregation_out_classes, bias=True).double())
                else:
                    if self.module_precision == 'float':
                        self.net_layers.append(nn.Linear(self.aggregation_out_classes, out_classes, bias=True).float())
                    else:
                        self.net_layers.append(nn.Linear(self.aggregation_out_classes, out_classes, bias=True).double())

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
            num_linear_layers = 0
            if 'linear_layers' in self.para.configs and self.para.configs['linear_layers'] > 0:
                num_linear_layers = self.para.configs['linear_layers']
            if num_linear_layers > 0:
                if i < len(self.net_layers) - num_linear_layers:
                    x = self.af(layer(x, pos))
                else:
                    if i < len(self.net_layers) - 1:
                        x = self.af(layer(x))
                    else:
                        x = self.out_af(layer(x))
            else:
                if i < len(self.net_layers) - 1:
                    x = self.af(layer(x, pos))
                else:
                    x = self.out_af(layer(x, pos))
        return x

    def return_info(self):
        return type(self)
