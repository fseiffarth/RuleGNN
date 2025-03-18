import torch_geometric
from torch import nn

from src.Architectures.BaseArchitecture import BaseGNN
from src.utils.GraphData import ShareGNNDataset
from src.utils.Parameters.Parameters import Parameters


class GCNGraph(BaseGNN):
    def initialize_graph_neural_network(self):
        gcn_network = torch_geometric.nn.GCNConv(in_channels=self.graph_data.num_node_features,
                                                 out_channels=self.graph_data.num_classes)

    def __init__(self, graph_data: ShareGNNDataset, para: Parameters, seed, device):
        super(GCNGraph, self).__init__(graph_data, para, seed, device)


