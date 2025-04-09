from src.Architectures.ShareGNN import ShareGNNLayers
import torch
import torch.nn as nn

from src.Architectures.ShareGNN.ShareGNNLayers import ShareGNNActivation

from src.Time.TimeClass import TimeClass
from src.utils.GraphData import ShareGNNDataset
from src.Architectures.ShareGNN.Parameters import Parameters


class ShareGNN(nn.Module):
    def __init__(self, graph_data: ShareGNNDataset, para: Parameters, seed, device):
        super(ShareGNN, self).__init__()
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
        input_features = self.graph_data.num_node_features
        output_features = input_features
        num_heads = 0
        for i, layer in enumerate(para.layers):
            prev_layer = (None if len(self.net_layers) == 0 else self.net_layers[-1])
            if prev_layer is not None:
                input_features = prev_layer.output_features
                num_heads = prev_layer.num_heads
                output_features = prev_layer.output_features
            if layer.layer_type == 'convolution':
                input_features = self.graph_data.num_node_features
                if i != 0 and self.para.run_config.config.get('use_feature_transformation', None) is not None:
                    input_features = self.para.run_config.config['use_feature_transformation'].get('out_dimension', 16)
                self.net_layers.append(
                    ShareGNNLayers.InvariantBasedMessagePassingLayer(layer_id=i,
                                                                    seed=seed + i,
                                                                    layer=layer,
                                                                    parameters=para,
                                                                    graph_data=self.graph_data,
                                                                    device=device,
                                                                     input_features=output_features,
                                                                     output_features=output_features).type(self.module_precision).requires_grad_(self.convolution_grad))


            elif layer.layer_type == 'aggregation':
                self.aggregation_out_dim = layer.layer_dict.get('out_dim', self.out_dim)
                self.net_layers.append(
                    ShareGNNLayers.InvariantBasedAggregationLayer(layer_id=i,
                                                                 seed=seed + i,
                                                                 layer=layer,
                                                                 parameters=para,
                                                                 out_dim=self.aggregation_out_dim,
                                                                 graph_data=self.graph_data,
                                                                 device=device,
                                                                  input_features=output_features,
                                                                  output_features=output_features).requires_grad_(self.aggregation_grad))
            elif layer.layer_type == 'linear':
                self.net_layers.append(ShareGNNLayers.ShareGNNLinear(layer, para, self.graph_data, num_heads=num_heads, input_features=input_features, output_features=output_features).type(self.module_precision))
            elif layer.layer_type == 'reshape':
                if isinstance(prev_layer, ShareGNNLayers.InvariantBasedAggregationLayer):
                    output_features = prev_layer.num_heads * prev_layer.output_features * prev_layer.output_dimension
                self.net_layers.append(ShareGNNLayers.ShareGNNReshapeLayer(layer, para, self.graph_data, num_heads=num_heads, input_features=input_features, output_features=output_features).type(self.module_precision))

        self.dropout = nn.Dropout(dropout)

        self.epoch = 0
        self.timer = TimeClass()




    def get_activation_function(self, key):
        if key in self.para.run_config.config and self.para.run_config.config[key] in ['None', 'Identity', 'identity', 'Id']:
            return ShareGNNActivation(nn.Identity())
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Relu', 'ReLU']:
            return ShareGNNActivation(nn.ReLU())
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LeakyRelu', 'LeakyReLU']:
            return ShareGNNActivation(nn.LeakyReLU())
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Tanh', 'tanh']:
            return ShareGNNActivation(nn.Tanh())
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Sigmoid', 'sigmoid']:
            return ShareGNNActivation(nn.Sigmoid())
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['Softmax', 'softmax']:
            return ShareGNNActivation(nn.Softmax(dim=0))
        elif key in self.para.run_config.config and self.para.run_config.config[key] in ['LogSoftmax', 'logsoftmax', 'log_softmax']:
            return ShareGNNActivation(nn.LogSoftmax(dim=0))
        else:
            # default is Identity but print a warning
            print(f'Activation function {key} not found. Using Identity activation function.')
            return ShareGNNActivation(nn.Identity())

    def forward(self, x, pos):
        for i, layer in enumerate(self.net_layers):
            x = layer(x, pos)
        return x

    def return_info(self):
        return type(self)




