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
        self.module_precision = 'double'
        dropout = self.para.dropout
        self.convolution_grad = self.para.run_config.config.get('convolution_grad', True)
        self.aggregation_grad = self.para.run_config.config.get('aggregation_grad', True)
        self.bias = self.para.run_config.config.get('bias', True)
        output_dimension = self.graph_data.output_feature_dimension
        self.module_precision = para.run_config.config.get('precision', 'float')
        self.aggregation_output_dimension = para.run_config.config.get('aggregation_out_features', output_dimension)
        self.channels = para.run_config.config.get('channels', 1)

        self.net_layers = nn.ModuleList()
        for i, layer in enumerate(para.layers):
            if i < len(para.layers) - 1:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        RuleGNNLayers.RuleConvolutionLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    bias=self.bias, precision=torch.float,
                                                    device=device).float().requires_grad_(
                            self.convolution_grad))
                else:
                    self.net_layers.append(
                        RuleGNNLayers.RuleConvolutionLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    bias=self.bias, precision=torch.double,
                                                    device=device).double().requires_grad_(
                            self.convolution_grad))
            else:
                if self.module_precision == 'float':
                    self.net_layers.append(
                        RuleGNNLayers.RuleAggregationLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    output_dimension=self.aggregation_output_dimension,
                                                    bias=self.bias,
                                                    precision=torch.float,
                                                           device=device).float().requires_grad_(
                            self.aggregation_grad))
                else:
                    self.net_layers.append(
                        RuleGNNLayers.RuleAggregationLayer(layer_id=i, seed=seed + i, layer_info=layer, parameters=para,
                                                    graph_data=self.graph_data,
                                                    output_dimension=self.aggregation_output_dimension,
                                                    bias=self.bias,
                                                    precision=torch.double, device=device).double().requires_grad_(
                            self.aggregation_grad))

        if 'linear_layers' in para.run_config.config and para.run_config.config['linear_layers'] > 0:
            for i in range(para.run_config.config['linear_layers']):
                if i < para.run_config.config['linear_layers'] - 1:
                    if self.module_precision == 'float':
                        self.net_layers.append(
                            nn.Linear(self.aggregation_output_dimension*self.graph_data.input_feature_dimension, self.aggregation_output_dimension*self.graph_data.input_feature_dimension, bias=True).float())
                    else:
                        self.net_layers.append(
                            nn.Linear(self.aggregation_output_dimension*self.graph_data.input_feature_dimension, self.aggregation_output_dimension*self.graph_data.input_feature_dimension, bias=True).double())
                else:
                    if self.module_precision == 'float':
                        self.net_layers.append(nn.Linear(self.aggregation_output_dimension*self.graph_data.input_feature_dimension, output_dimension, bias=True).float())
                    else:
                        self.net_layers.append(nn.Linear(self.aggregation_output_dimension*self.graph_data.input_feature_dimension, output_dimension, bias=True).double())

        elif self.channels*self.aggregation_output_dimension * self.graph_data.input_feature_dimension != output_dimension:
            if self.module_precision == 'float':
                self.net_layers.append(
                    nn.Linear(self.channels*self.aggregation_output_dimension * self.graph_data.input_feature_dimension,
                              output_dimension, bias=True).float())
            else:
                self.net_layers.append(
                    nn.Linear(self.channels*self.aggregation_output_dimension * self.graph_data.input_feature_dimension,
                              output_dimension, bias=True).double())

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
            elif self.channels*self.aggregation_output_dimension * self.graph_data.input_feature_dimension != self.graph_data.output_feature_dimension:
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
