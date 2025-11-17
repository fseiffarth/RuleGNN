import torch

from src.Architectures.Layers.FrameworkLayers import FrameworkLayers, GNNConvLayer
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
                output_features = layer.layer_dict.get('out_features', 16)
                layer_args = {
                    'in_features': current_feature_dimension,
                    'out_features': output_features,
                    'bias': layer.layer_dict.get('bias', True),
                    'dtype': self.module_precision
                }
                self.net_layers.append(GNNFrameworkLayers.LinearLayer(layer_args))
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
            elif layer.layer_type in [LayerTypes.GCN_CONVOLUTION.value,
                                      LayerTypes.GAT_CONVOLUTION.value,
                                      LayerTypes.GATv2_CONVOLUTION.value,
                                      LayerTypes.GIN_CONVOLUTION.value,
                                      LayerTypes.SAGE_CONVOLUTION.value]:
                layer_args = layer.layer_dict
                layer_args['in_channels'] = current_feature_dimension
                current_feature_dimension = layer_args['out_channels']
                # GNN specific layers
                if layer.layer_type == LayerTypes.GCN_CONVOLUTION.value:
                    self.net_layers.append(GNNFrameworkLayers.GCNConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GAT_CONVOLUTION.value:
                    self.net_layers.append(GNNFrameworkLayers.GATConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GATv2_CONVOLUTION.value:
                    self.net_layers.append(GNNFrameworkLayers.GATv2Conv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.GIN_CONVOLUTION.value:
                    self.net_layers.append(GNNFrameworkLayers.GINConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))
                elif layer.layer_type == LayerTypes.SAGE_CONVOLUTION.value:
                    self.net_layers.append(GNNFrameworkLayers.SAGEConv(layer_args).type(self.module_precision).requires_grad_(self.convolution_grad))

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
            elif layer.layer_type == LayerTypes.BATCH_NORM.value:
                layer_args = {'batch_norm': True, 'in_channels': current_feature_dimension}
                self.net_layers.append(GNNFrameworkLayers.BatchNormLayer(layer_args).type(self.module_precision))
                # BatchNorm does not change feature dimension
            else:
                raise ValueError(f'Layer type {layer.layer_type} not recognized in OrdinaryGNN')
        self.dropout = nn.Dropout(dropout)

        self.epoch = 0
        self.timer = TimeClass()

    def forward(self, batch_data, *args, **kwargs):
        x = batch_data.x
        representation_list = []
        for i, layer in enumerate(self.net_layers):
            x = layer(x, batch_data)
            #if isinstance(layer, GNNConvLayer):
            #    representation_list.append(x)
            #    if i == len(self.net_layers) - 1 or not isinstance(self.net_layers[i + 1], GNNConvLayer):
            #        x = torch.squeeze(torch.mean(torch.stack(representation_list), 0, True), 0)

        return x

    def return_info(self):
        return type(self)
