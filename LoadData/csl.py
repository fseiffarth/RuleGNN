import networkx as nx
import torch
import torch_geometric
from matplotlib import pyplot as plt

from torch_geometric.datasets import GNNBenchmarkDataset
import torch
from torch_geometric.data import InMemoryDataset, download_url

from GraphData import NodeLabeling, EdgeLabeling
from GraphData.GraphData import GraphData
import RuleFunctions.Rules as rule

from torch_geometric.utils import to_networkx


class PrepareCSL(InMemoryDataset):
    """This class is used to download and process the CSL (Circular Skip Link) dataset from the
    Benchmarking GNNs paper. The dataset contains 10 isomorphism classes of regular graphs that
    must be classified.
    """

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        pass


        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        # Read data into huge `Data` list.
        data_list = [...]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        # torch.save(self.collate(data_list), self.processed_paths[0])

    # def download(self):
    #     # Instantiating this will download and process the graph dataset.
    #     GNNBenchmarkDataset(self.raw_dir, 'CSL')

    def get_graphs(self, with_distances=False, with_cycles=False):
        nx_graphs = []
        original_source = -1
        # iterate over edge_index in self.data and add edges to nx_graph
        for edge in self.data.edge_index.T:
            source = edge[0].item()
            target = edge[1].item()
            if nx_graphs == [] or source < original_source:
                nx_graphs.append(nx.Graph())
            original_source = source
            nx_graphs[-1].add_edge(source, target)
        labels = [x.item() for x in self.data["y"]]

        self.graphs = GraphData()
        self.graphs.graph_labels = labels
        self.graphs.graphs = nx_graphs
        self.graphs.num_graphs = len(nx_graphs)
        self.graphs.num_classes = len(set(labels))
        self.graphs.graph_db_name = "CSL"
        self.graphs.inputs = [torch.ones(g.number_of_nodes()).double() for g in nx_graphs]
        if with_distances:
            self.graphs.distance_list = []
            for graph in nx_graphs:
                self.graphs.distance_list.append(dict(nx.all_pairs_shortest_path_length(graph, cutoff=6)))
        if with_cycles:
            self.graphs.cycle_list = []
            for graph in nx_graphs:
                self.graphs.cycle_list.append(rule.generate_cycle_list(graph))

        # get one hot labels from graph labels
        self.graphs.one_hot_labels = torch.nn.functional.one_hot(torch.tensor(labels)).double()
        # set the labels
        self.graphs.primary_node_labels, self.graphs.primary_edge_labels = self.graphs.add_node_labels(node_labeling_method=NodeLabeling.standard_node_labeling, edge_labeling_method=EdgeLabeling.standard_edge_labeling)
        pass


    def load_dataset(self):
        """Load the dataset from here and process it if it doesn't exist"""
        print("Loading dataset from disk...")
        data = torch_geometric.data.Data(self.processed_dir)
        data = data[0]
        graph = nx.Graph()
        # iterate over edge_index
        labels = list(data["y"])
        # generate nx graphs from data
        print("Dataset loaded.")

        return data

    def graph_data(self, with_distances=False, with_cycles=False):
        self.get_graphs(with_distances, with_cycles)
        """Return the processed graph data"""
        return self.graphs


def main():
    PrepareCSL("Datasets/CSL")


if __name__ == "__main__":
    main()
