from abc import abstractmethod, ABC

import torch
import torch_geometric
from torch._C.cpp import nn

from src.Preprocessing.GraphData.GraphData import ShareGNNDataset


class FrameworkLayers(torch.nn.Module, ABC):
    """
    A base class for the layers in our GNN framework.
    """
    def __init__(self, layer_args, device='cpu'):
        super(FrameworkLayers, self).__init__()
        self.layer_args = layer_args
        # Whether to use residual connections in this layer
        self.residual = layer_args.get('residual', False)
        # Whether to use batch normalization in this layer
        self.batch_norm = layer_args.get('batch_norm', False)
        if self.batch_norm:
            self.batch_norm_args = {
                'in_channels': layer_args.get('in_channels', None),
                'eps': layer_args.get('batch_norm_eps', 1e-5),
                'momentum': layer_args.get('batch_norm_momentum', 0.1),
                'affine': layer_args.get('batch_norm_affine', True),
                'track_running_stats': layer_args.get('batch_norm_track_running_stats', True),
                'allow_single_element': layer_args.get('batch_norm_allow_single_element', False),
            }
            self.batch_norm_layer = torch_geometric.nn.BatchNorm(**self.batch_norm_args)
        # Dropout rate for this layer
        self.dropout = layer_args.get('dropout', 0.0)
        # Device to run the layer on
        self.device = device
        # Activation function for this layer
        self.activation = torch.nn.Identity()
        if 'activation' in layer_args:
            self.activation = eval(layer_args['activation'])

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        pass

class GNNConvLayer(FrameworkLayers):
    def __init__(self, layer_args):
        super(GNNConvLayer, self).__init__(layer_args)

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return node_representation

class GCNConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GCNConv, self).__init__(layer_args)
        gcn_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'improved': layer_args.get('improved', False),
            'cached': layer_args.get('cached', False),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'normalize': layer_args.get('normalize', True),
            'bias': layer_args.get('bias', True)
        }
        self.layer = torch_geometric.nn.GCNConv(**gcn_args)

    def forward(self,node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation

class GATConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GATConv, self).__init__(layer_args)
        self.gat_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'heads': layer_args.get('heads', 1),
            'concat': layer_args.get('concat', False),
            'negative_slope': layer_args.get('negative_slope', 0.2),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'edge_dim': layer_args.get('edge_dim', None),
            'fill_value': layer_args.get('fill_value', 'mean'),
            'bias': layer_args.get('bias', True),
        }
        self.merge_heads = layer_args.get('merge_heads', True)
        self.layer = torch_geometric.nn.GATConv(**self.gat_args)
        self.linear_merge_heads = torch.nn.Linear(self.gat_args['out_channels'] * self.gat_args['heads'], self.gat_args['out_channels']) if self.gat_args['concat'] else torch.nn.Linear(self.gat_args['out_channels'], self.gat_args['out_channels'])


    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.merge_heads and self.gat_args['concat'] and self.gat_args['heads'] > 1:
            node_representation = self.linear_merge_heads(node_representation)
        if self.batch_norm:
            if self.merge_heads:
                node_representation = self.batch_norm_layer(node_representation)
            else: # apply batch norm to each head separately
                node_representation = self.batch_norm_layer(node_representation.view(-1, self.gat_args['out_channels'])).view(-1, self.gat_args['out_channels'] * self.gat_args['heads'])
        node_representation = self.activation(node_representation)
        if self.residual:
            if self.merge_heads:
                node_representation = node_representation + x
            else: # add residual to each head separately
                node_representation = node_representation + x.repeat(1, self.gat_args['heads'])
        if self.dropout > 0:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation

class GATv2Conv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GATv2Conv, self).__init__(layer_args)
        self.gatv2_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'heads': layer_args.get('heads', 1),
            'concat': layer_args.get('concat', False),
            'negative_slope': layer_args.get('negative_slope', 0.2),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'edge_dim': layer_args.get('edge_dim', None),
            'fill_value': layer_args.get('fill_value', 'mean'),
            'bias': layer_args.get('bias', True),
            'share_weights': layer_args.get('share_weights', False),
        }
        self.merge_heads = layer_args.get('merge_heads', True)
        self.layer = torch_geometric.nn.GATv2Conv(**self.gatv2_args)
        if self.merge_heads:
            self.linear_merge_heads = torch.nn.Linear(self.gatv2_args['out_channels'] * self.gatv2_args['heads'], self.gatv2_args['out_channels']) if self.gatv2_args['concat'] else torch.nn.Linear(self.gatv2_args['out_channels'], self.gatv2_args['out_channels'])

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.merge_heads and self.gatv2_args['concat'] and self.gatv2_args['heads'] > 1:
            node_representation = self.linear_merge_heads(node_representation)
        if self.batch_norm:
            if self.merge_heads:
                node_representation = self.batch_norm_layer(node_representation)
            else: # apply batch norm to each head separately
                node_representation = self.batch_norm_layer(node_representation.view(-1, self.gatv2_args['out_channels'])).view(-1, self.gatv2_args['out_channels'] * self.gatv2_args['heads'])
        node_representation = self.activation(node_representation)
        if self.residual:
            if self.merge_heads:
                node_representation = node_representation + x
            else: # add residual to each head separately
                node_representation = node_representation + x.repeat(1, self.gatv2_args['heads'])
        if self.dropout > 0:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation

class SAGEConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(SAGEConv, self).__init__(layer_args)
        self.sage_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'aggr': layer_args.get('aggr', 'mean'),  # "mean", "max", "add"
            'normalize': layer_args.get('normalize', False),
            'root_weight': layer_args.get('root_weight', True),
            'project': layer_args.get('project', False),
            'bias': layer_args.get('bias', True),
        }
        self.layer = torch_geometric.nn.SAGEConv(**self.sage_args)


    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation

class GINConv(GNNConvLayer):
    def __init__(self, layer_args):
        super(GINConv, self).__init__(layer_args)
        gin_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'eps': layer_args.get('eps', 0.0),
            'train_eps': layer_args.get('train_eps', False),
            'bias': layer_args.get('bias', True),
        }
        self.layer = torch_geometric.nn.GINConv(**gin_args)

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        x = node_representation
        node_representation = self.layer(node_representation, batch_data.edge_index)
        if self.batch_norm:
            node_representation = self.batch_norm_layer(node_representation)
        node_representation = self.activation(node_representation)
        if self.residual:
            node_representation = node_representation + x
        if self.dropout > 0:
            node_representation = torch.nn.Dropout(self.dropout)(node_representation)
        return node_representation

class GlobalPooling(FrameworkLayers):
    def __init__(self, layer_args):
        super(GlobalPooling, self).__init__(layer_args)
        self.mode = layer_args.get('mode', 'mean')
        self.pooling_function = None
        if self.mode == 'mean':
            self.pooling_function = torch_geometric.nn.global_mean_pool
        elif self.mode == 'max':
            self.pooling_function = torch_geometric.nn.global_max_pool
        elif self.mode == 'sum':
            self.pooling_function = torch_geometric.nn.global_add_pool
        else:
            raise ValueError(f"Unsupported pooling mode: {self.mode}")
        self.name = f"Global {self.mode.capitalize()} Pooling"


    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.activation(self.pooling_function(node_representation, batch_data.batch))

class LinearLayer(FrameworkLayers):
    def __init__(self, layer_args):
        super(LinearLayer, self).__init__(layer_args)
        self.layer = torch.nn.Linear(**layer_args)
        self.name = "Linear Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.activation(self.layer(node_representation))

class ActivationLayer(FrameworkLayers):
    def __init__(self, layer_args):
        activation_function = layer_args.get('activation_function', torch.nn.Identity())
        super(ActivationLayer, self).__init__(layer_args)
        self.activation_function = activation_function
        self.name = "Activation Function"

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.activation_function(node_representation)


class DropoutLayer(FrameworkLayers):
    def __init__(self, layer_args):
        p = layer_args.get('p', 0.5)
        super(DropoutLayer, self).__init__(layer_args)
        self.dropout = torch.nn.Dropout(p)
        self.name = "Dropout Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.dropout(node_representation)

