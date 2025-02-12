from torch.cuda import graph

from src.Architectures.RuleGNN import RuleGNNLayers
import torch
import torch.nn as nn
from src.utils import GraphData

from src.Time.TimeClass import TimeClass
from src.utils.GraphData import RuleGNNDataset
from src.utils.Parameters.Parameters import Parameters


class RuleGNN(nn.Module):
    def __init__(self, graph_data: RuleGNNDataset, para: Parameters, seed, device):
        super(RuleGNN, self).__init__()
        self.graph_data = graph_data
        self.para = para
        self.print_weights = self.para.net_print_weights
        dropout = self.para.dropout
        self.convolution_grad = self.para.run_config.config.get('convolution_grad', True)
        self.aggregation_grad = self.para.run_config.config.get('aggregation_grad', True)
        self.out_dim = self.graph_data.num_classes
        precision = para.run_config.config.get('precision', 'float')
        self.module_precision = torch.float
        if precision == 'double':
            self.module_precision = torch.double

        self.aggregation_out_dim = 0

        nn.Sequential(

        )

        # Define the layers
        self.net_layers = nn.ModuleList()
        for i, layer in enumerate(para.layers):
            if layer.layer_type == 'convolution':
                self.net_layers.append(
                    RuleGNNLayers.RuleConvolutionLayer(layer_id=i,
                                                       seed=seed + i,
                                                       layer=layer,
                                                       parameters=para,
                                                       graph_data=self.graph_data,
                                                       device=device).type(self.module_precision).requires_grad_(self.convolution_grad))
                # Concatenate multiple heads using a linear layer
                if layer.num_heads() > 1 and layer.layer_dict.get('concatenate_heads', True):
                    self.net_layers.append(RuleGNNLayers.RuleGNNConcatenate(layer.num_heads(), bias=True).type(
                        self.module_precision).requires_grad_(True))

            elif layer.layer_type == 'aggregation':
                self.aggregation_out_dim = layer.layer_dict.get('out_dim', self.out_dim)
                self.net_layers.append(
                    RuleGNNLayers.RuleAggregationLayer(layer_id=i,
                                                       seed=seed + i,
                                                       layer=layer,
                                                       parameters=para,
                                                       out_dim=self.aggregation_out_dim,
                                                       graph_data=self.graph_data,
                                                       device=device).type(self.module_precision).requires_grad_(self.aggregation_grad))
            if i == len(para.layers) - 1 and self.para.run_config.config.get('final_layer', True):
                # Add a final linear layer to get the output dimension
                if layer.num_heads() * self.aggregation_out_dim * self.graph_data.num_node_features != self.out_dim:
                    self.net_layers.append(RuleGNNLayers.RuleGNNLinear(layer.num_heads() * self.aggregation_out_dim * self.graph_data.num_node_features, self.out_dim, bias=True).type(self.module_precision).requires_grad_(True))


        if 'final_linear_layers' in para.run_config.config and len(para.run_config.config['final_linear_layers']) > 0:
            for layer in para.run_config.config['final_linear_layers']:
                input_dimension = layer.get('input_dimension', max(self.aggregation_out_dim,1) * self.graph_data.num_node_features)
                output_dimension = layer.get('output_dimension', self.out_dim)
                bias = layer.get('bias', True)
                self.net_layers.append(RuleGNNLayers.RuleGNNLinear(input_dimension, output_dimension, bias=bias).type(self.module_precision).requires_grad_(True))

        if 'linear_layers' in para.run_config.config and para.run_config.config['linear_layers'] > 0:
            for i in range(para.run_config.config['linear_layers']):
                if i < para.run_config.config['linear_layers'] - 1:
                    self.net_layers.append(RuleGNNLayers.RuleGNNLinear(self.aggregation_out_dim * self.graph_data.num_node_features, self.aggregation_out_dim * self.graph_data.num_node_features, bias=True).type(self.module_precision).requires_grad_(True))
                else:
                    self.net_layers.append(RuleGNNLayers.RuleGNNLinear(self.aggregation_out_dim * self.graph_data.num_node_features, self.out_dim, bias=True).type(self.module_precision).requires_grad_(True))


        self.dropout = nn.Dropout(dropout)
        self.af = self.get_activation_function('activation')
        self.out_af = self.get_activation_function('output_activation')

        self.epoch = 0
        self.timer = TimeClass()

    def get_activation_function(self, key):
        if key in self.para.run_config.config and self.para.run_config.config[key] in ['None', 'Identity', 'identity', 'Id']:
            return nn.Identity()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Relu', 'ReLU']:
            return nn.ReLU()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LeakyRelu', 'LeakyReLU']:
            return nn.LeakyReLU()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Tanh', 'tanh']:
            return nn.Tanh()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Sigmoid', 'sigmoid']:
            return nn.Sigmoid()
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Softmax', 'softmax']:
            return nn.Softmax(dim=0)
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LogSoftmax', 'logsoftmax', 'log_softmax']:
            return nn.LogSoftmax(dim=0)
        else:
            raise ValueError(f'Activation function {key} not recognized')

    def forward(self, x, pos):
        for i, layer in enumerate(self.net_layers):
            if i == len(self.net_layers) - 1:
                    x = self.out_af(layer(x, pos))
            else:
                    x = self.af(layer(x, pos))
                    x = self.dropout(x)
        return x

    def return_info(self):
        return type(self)




