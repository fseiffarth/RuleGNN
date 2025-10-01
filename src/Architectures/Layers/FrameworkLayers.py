from abc import abstractmethod, ABC

import torch
import torch_geometric
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

    def forward(self,node_representation:torch.Tensor, data: ShareGNNDataset, *args, **kwargs):
        return self.layer(node_representation, data.edge_index)

class MeanAggregation(FrameworkLayers):
    def __init__(self, layer_args):
        super(MeanAggregation, self).__init__(layer_args)

    def forward(self, node_representation:torch.Tensor, data: ShareGNNDataset, *args, **kwargs):
        return torch_geometric.nn.global_mean_pool(node_representation, data.slices['x'])

