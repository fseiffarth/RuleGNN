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

    @abstractmethod
    def forward(self, node_representation:torch.Tensor, data: ShareGNNDataset, *args, **kwargs):
        pass

class GCNConv(FrameworkLayers):
    def __init__(self, layer_args):
        super(GCNConv, self).__init__(layer_args)
        self.layer = torch_geometric.nn.GCNConv(**layer_args)

    def forward(self,node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.layer(node_representation, batch_data.edge_index)

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
        return self.pooling_function(node_representation, batch_data.batch)

class LinearLayer(FrameworkLayers):
    def __init__(self, layer_args):
        super(LinearLayer, self).__init__(layer_args)
        self.layer = torch.nn.Linear(**layer_args)
        self.name = "Linear Layer"

    def forward(self, node_representation:torch.Tensor, batch_data: ShareGNNDataset, *args, **kwargs):
        return self.layer(node_representation)

class ActivationLayer(nn.Module):
    def __init__(self, layer_args):
        activation_function = layer_args.get('activation_function', torch.nn.Identity())
        super(ActivationLayer, self).__init__()
        self.activation_function = activation_function
        self.name = "Activation Function"

    def forward(self, x: torch.Tensor, pos:int=None):
        return self.activation_function(x)

