from torch.cuda import graph

from src.Architectures.RuleGNN import RuleGNNLayers
import torch
import torch.nn as nn
from src.utils import GraphData

from src.Time.TimeClass import TimeClass
from src.utils.Parameters.Parameters import Parameters


class RuleGNN(nn.Module):
    def __init__(self, graph_data: GraphData, para: Parameters, seed, device):
        super(RuleGNN, self).__init__()
        self.graph_data = graph_data
        self.para = para
        self.print_weights = self.para.net_print_weights
        dropout = self.para.dropout
        self.convolution_grad = self.para.run_config.config.get('convolution_grad', True)
        self.aggregation_grad = self.para.run_config.config.get('aggregation_grad', True)
        self.bias = self.para.run_config.config.get('bias', True)
        out_dim = self.graph_data.output_feature_dimension
        precision = para.run_config.config.get('precision', 'float')
        self.module_precision = torch.float
        if precision == 'double':
            self.module_precision = torch.double

        self.aggregation_out_dim = 0
        self.channels = para.run_config.config.get('channels', 1)
        self.channels = max(self.channels, graph_data.input_channels)
        if self.channels % graph_data.input_channels == 0:
            if self.channels > graph_data.input_channels:
                num_stacks = self.channels // graph_data.input_channels
                # modify the input channels to match the number of channels
                for i, input_vector in enumerate(graph_data.input_data):
                    # stack input_vector num_stacks times along the first dimension
                    graph_data.input_vectors[i] = input_vector.repeat(num_stacks, 1)

        else:
            raise ValueError('If number of channels is larger than input channels, it must be a multiple of input channels')

        self.net_layers = nn.ModuleList()
        for i, layer in enumerate(para.layers):
            if i < len(para.layers) - 1:
                self.net_layers.append(
                    RuleGNNLayers.RuleConvolutionLayer(layer_id=i,
                                                       seed=seed + i,
                                                       layer_info=layer,
                                                       parameters=para,
                                                       bias=self.bias,
                                                       out_channels=self.channels,
                                                       graph_data=self.graph_data,
                                                       device=device).type(self.module_precision).requires_grad_(self.convolution_grad))
            else:
                self.aggregation_out_dim = layer.layer_dict.get('out_dim', out_dim)
                self.net_layers.append(
                    RuleGNNLayers.RuleAggregationLayer(layer_id=i,
                                                       seed=seed + i,
                                                       layer_info=layer,
                                                       parameters=para,
                                                       out_channels=self.channels,
                                                       out_dim=self.aggregation_out_dim,
                                                       graph_data=self.graph_data,
                                                       bias=self.bias,
                                                       device=device).type(self.module_precision).requires_grad_(self.aggregation_grad))


        if 'linear_layers' in para.run_config.config and para.run_config.config['linear_layers'] > 0:
            for i in range(para.run_config.config['linear_layers']):
                if i < para.run_config.config['linear_layers'] - 1:
                    self.net_layers.append(nn.Linear(self.aggregation_out_dim * self.graph_data.input_feature_dimension, self.aggregation_out_dim * self.graph_data.input_feature_dimension, bias=True).type(self.module_precision).requires_grad_(True))
                else:
                    self.net_layers.append(nn.Linear(self.aggregation_out_dim * self.graph_data.input_feature_dimension, out_dim, bias=True).type(self.module_precision).requires_grad_(True))

        elif self.channels*self.aggregation_out_dim * self.graph_data.input_feature_dimension != out_dim:
                self.net_layers.append(
                    nn.Linear(self.channels * self.aggregation_out_dim * self.graph_data.input_feature_dimension,
                              out_dim, bias=True).type(self.module_precision).requires_grad_(True))

        self.dropout = nn.Dropout(dropout)
        if 'activation' in para.run_config.config and para.run_config.config['activation'] == 'None':
            self.af = nn.Identity()
        elif 'activation' in para.run_config.config and para.run_config.config['activation'] == 'Relu':
            self.af = nn.ReLU()
        elif 'activation' in para.run_config.config and para.run_config.config['activation'] == 'LeakyRelu':
            self.af = nn.LeakyReLU()
        else:
            self.af = nn.Tanh()

        if 'output_activation' in para.run_config.config and para.run_config.config['output_activation'] == 'None':
            self.out_af = nn.Identity()
        elif 'output_activation' in para.run_config.config and para.run_config.config['output_activation'] == 'Relu':
            self.out_af = nn.ReLU()
        elif 'output_activation' in para.run_config.config and para.run_config.config['output_activation'] == 'LeakyRelu':
            self.out_af = nn.LeakyReLU()
        else:
            self.out_af = nn.Tanh()

        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, x, pos):
        for i, layer in enumerate(self.net_layers):
            num_linear_layers = 0
            if 'linear_layers' in self.para.run_config.config and self.para.run_config.config['linear_layers'] > 0:
                num_linear_layers = self.para.run_config.config['linear_layers']
            elif self.channels*self.aggregation_out_dim * self.graph_data.input_feature_dimension != self.graph_data.output_feature_dimension:
                num_linear_layers = 1
            if num_linear_layers > 0:
                if i < len(self.net_layers) - num_linear_layers:
                    x = self.dropout(x)
                    x = self.af(layer(x, pos))
                else:
                    x = self.dropout(x)
                    # flatten the input
                    x = torch.flatten(x)
                    if i < len(self.net_layers) - 1:
                        x = self.af(layer(x))
                    else:
                        x = self.out_af(layer(x))
            else:
                if i < len(self.net_layers) - 1:
                    x = self.dropout(x)
                    x = self.af(layer(x, pos))
                    x = self.dropout(x)
                else:
                    x = self.dropout(x)
                    x = self.out_af(layer(x, pos))
        return torch.flatten(x)

    def return_info(self):
        return type(self)




