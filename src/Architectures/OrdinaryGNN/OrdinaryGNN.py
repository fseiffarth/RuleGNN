import torch

from src.Architectures.Layers.FrameworkLayers import FrameworkLayers
from src.Architectures.ShareGNN import ShareGNNLayers
from src.Architectures.ShareGNN.Parameters import Parameters
from src.Customize.LayerTypes import LayerTypes
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
import torch.nn as nn
import torch_geometric
import src.Architectures.Layers.FrameworkLayers as GNNFrameworkLayers
from src.Time.TimeClass import TimeClass


class OrdinaryGNN(torch.nn.Module):
    def __init__(self, graph_data: ShareGNNDataset, para: Parameters, seed, device):

        super(OrdinaryGNN, self).__init__()
        self.graph_data = graph_data
        self.para = para
        self.config_parameters = para.run_config.config
        self.print_weights = self.para.net_print_weights
        dropout = self.para.dropout
        self.convolution_grad =self.config_parameters.get('convolution_grad', True)
        self.aggregation_grad =self.config_parameters.get('aggregation_grad', True)
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
        num_heads = 0
        current_feature_dimension = input_features
        for i, layer in enumerate(para.layers):
            if layer.layer_type == 'invariant_based_convolution':
                if i != 0 and self.config_parameters.get('use_feature_transformation', None) is not None:
                    input_features = self.config_parameters['use_feature_transformation'].get('out_dimension', 16)
                self.net_layers.append(
                    ShareGNNLayers.InvariantBasedMessagePassingLayer(layer_id=i,
                                                                    seed=seed,
                                                                    layer=layer,
                                                                    parameters=para,
                                                                    graph_data=self.graph_data,
                                                                    device=device,
                                                                     input_features=current_feature_dimension,
                                                                     output_features=current_feature_dimension).type(self.module_precision).requires_grad_(self.convolution_grad))

            elif layer.layer_type == 'invariant_based_aggregation':
                self.aggregation_out_dim = layer.layer_dict.get('out_dim', self.out_dim)
                self.net_layers.append(
                    ShareGNNLayers.InvariantBasedAggregationLayer(layer_id=i,
                                                                 seed=seed,
                                                                 layer=layer,
                                                                 parameters=para,
                                                                 out_dim=self.aggregation_out_dim,
                                                                 graph_data=self.graph_data,
                                                                 device=device,
                                                                  input_features=current_feature_dimension,
                                                                  output_features=current_feature_dimension).requires_grad_(self.aggregation_grad))
            elif layer.layer_type == 'linear':
                layer_args = {
                    'in_features': current_feature_dimension,
                    'out_features': layer.layer_dict.get('out_features', 16),
                    'bias': layer.layer_dict.get('bias', True)
                }
                self.net_layers.append(OrdinaryGNN.LinearLayer(**layer_args))
                self.net_layers.append(ShareGNNLayers.ShareGNNLinear(layer_id=i,
                                                                     seed=seed,
                                                                     layer=layer,
                                                                     parameters=para,
                                                                     graph_data=self.graph_data,
                                                                     num_heads=num_heads,
                                                                     input_features=input_features,
                                                                     output_features=output_features).type(self.module_precision)).requires_grad_()
                current_feature_dimension = output_features
            elif layer.layer_type == 'reshape':
                if isinstance(prev_layer, ShareGNNLayers.InvariantBasedAggregationLayer):
                    output_features = prev_layer.num_heads * prev_layer.output_features * prev_layer.output_dimension
                self.net_layers.append(ShareGNNLayers.ShareGNNReshapeLayer(layer_id=i,
                                                                           seed=seed,
                                                                           layer=layer,
                                                                           parameters=para,
                                                                           graph_data=self.graph_data,
                                                                           num_heads=num_heads,
                                                                           input_features=input_features,
                                                                           output_features=output_features).type(self.module_precision))
                current_feature_dimension = output_features
            elif layer.layer_type == 'layer_norm':
                self.net_layers.append(ShareGNNLayers.ShareGNNLayerNorm(layer_id=i,
                                                                        num_heads=num_heads,
                                                                        input_features=input_features,
                                                                        output_features=output_features).type(self.module_precision))
                current_feature_dimension = output_features
            elif layer.layer_type == 'gcn_convolution':
                gcn_args = {'in_channels': current_feature_dimension,
                            'out_channels': layer.layer_dict.get('out_channels', 16),
                            'improved': layer.layer_dict.get('improved', False),
                            'cached': layer.layer_dict.get('cached', False),
                            'add_self_loops': layer.layer_dict.get('add_self_loops', None),
                            'normalize': layer.layer_dict.get('normalize', True),
                            'bias': layer.layer_dict.get('bias', True)}
                current_feature_dimension = gcn_args['out_channels']
                self.net_layers.append(GNNFrameworkLayers.GCNConv(gcn_args).type(self.module_precision).requires_grad_(self.convolution_grad))
            elif layer.layer_type == LayerTypes.GLOBAL_POOLING.value:
                layer_args = {'mode': layer.layer_dict.get('mode', 'mean')}
                self.net_layers.append(GNNFrameworkLayers.GlobalPooling(layer_args).type(self.module_precision).requires_grad_(self.aggregation_grad))
            elif layer.layer_type == LayerTypes.DROPOUT.value:
                layer_args = {'p': layer.layer_dict.get('p', 0.5)}
                self.net_layers.append(GNNFrameworkLayers.DropoutLayer(layer_args))
                # Dropout does not change feature dimension
            elif layer.layer_type == LayerTypes.ACTIVATION.value:
                layer_args = {'activation_function': layer.layer_dict.get('activation_function', torch.nn.ReLU())}
                self.net_layers.append(GNNFrameworkLayers.ActivationLayer(layer_args))
                # Activation does not change feature dimension
            else:
                raise ValueError(f'Layer type {layer.layer_type} not recognized in OrdinaryGNN')
        self.dropout = nn.Dropout(dropout)

        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, data_batch, *args, **kwargs):
        node_representation = data_batch.x
        for i, layer in enumerate(self.net_layers):
            node_representation = layer(node_representation, data_batch)
        return node_representation

    def return_info(self):
        return type(self)
