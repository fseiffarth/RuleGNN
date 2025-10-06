from abc import abstractmethod, ABC

import torch
import torch_geometric
from torch._C.cpp import nn

from src.Preprocessing.GraphData.GraphData import ShareGNNDataset


class FrameworkLayers(torch.nn.Module, ABC):
    """
    A base class for the layers in our GNN framework.
    """
    def __init__(self, layer_args):
        super(FrameworkLayers, self).__init__()
        self.layer_args = layer_args
        self.activation = torch.nn.Identity()
        if 'activation' in layer_args:
            self.activation = eval(layer_args['activation'])

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        pass

class GCNConv(FrameworkLayers):
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
        return  self.activation(self.layer(node_representation, batch_data.edge_index))

class GATConv(FrameworkLayers):
    def __init__(self, layer_args):
        super(GATConv, self).__init__(layer_args)
        gat_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'heads': layer_args.get('heads', 1),
            'concat': layer_args.get('concat', True),
            'negative_slope': layer_args.get('negative_slope', 0.2),
            'add_self_loops': layer_args.get('add_self_loops', True),
            'edge_dim': layer_args.get('edge_dim', None),
            'fill_value': layer_args.get('fill_value', 'mean'),
            'bias': layer_args.get('bias', True),
            'residual': layer_args.get('residual', True),
        }
        self.layer = torch_geometric.nn.GATConv(**gat_args)

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.activation(self.layer(node_representation, batch_data.edge_index))

class SAGEConv(FrameworkLayers):
    def __init__(self, layer_args):
        super(SAGEConv, self).__init__(layer_args)
        sage_args = {
            'in_channels': layer_args.get('in_channels'),
            'out_channels': layer_args.get('out_channels'),
            'aggr': layer_args.get('aggr', 'mean'),  # "mean", "max", "add"
            'normalize': layer_args.get('normalize', False),
            'root_weight': layer_args.get('root_weight', True),
            'project': layer_args.get('project', False),
            'bias': layer_args.get('bias', True),
        }
        self.layer = torch_geometric.nn.SAGEConv(**sage_args)

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.activation(self.layer(node_representation, batch_data.edge_index))

class GINConv(FrameworkLayers):
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
        return self.activation(self.layer(node_representation, batch_data.edge_index))

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

