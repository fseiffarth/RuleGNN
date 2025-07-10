# abstract class for graph data preprocessing
import abc

import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import ZINC


class GraphDataPreprocessing(abc.ABC):
    def __init__(self, name, tmp_dir="/tmp"):
        self.name = name
        self.tmp_dir = tmp_dir
        self.processed_dataset = None
        self.slices = None
        self.sizes =  {'num_edge_attributes': None,
                         'num_edge_labels': None,
                         'num_node_attributes': None,
                         'num_node_labels': None
                }


    @abc.abstractmethod
    def preprocess(self, *args, **kwargs):
        """
        Abstract method to preprocess the raw dataset.
        This method should be implemented by subclasses to perform specific preprocessing tasks.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        return NotImplementedError("Subclasses should implement this method.")


    def set_sizes(self):
        if self.processed_dataset is None:
            raise ValueError("Processed dataset is not set. Please run preprocess() first.")

        self.sizes = {'num_edge_attributes': self.processed_dataset.edge_attributes.shape[-1],
                 'num_edge_labels': len(torch.unique(self.processed_dataset.primary_edge_labels)),
                 'num_node_attributes': self.processed_dataset.node_attributes.shape[-1],
                 'num_node_labels': len(torch.unique(self.processed_dataset.primary_node_labels))
                 }


class ZINCGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the ZINC dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        subset = True
        if self.name in ['ZINC-full', 'ZINC-Full', 'ZINCFull', 'ZINC-250k']:
            subset = False
        train_data = ZINC(root=self.tmp_dir, subset=subset, split='train')
        validation_data = ZINC(root=self.tmp_dir, subset=subset, split='val')
        test_data = ZINC(root=self.tmp_dir, subset=subset, split='test')
        # merge train_data._data, validation_data._data and test_data._data
        all_data = torch_geometric.data.InMemoryDataset.collate(
            [train_data._data, validation_data._data, test_data._data])

        self.processed_dataset = all_data[0]

        # merge the slices
        self.slices = dict()
        for key in train_data.slices.keys():
            validation_data.slices[key] += train_data.slices[key][-1]
            test_data.slices[key] += validation_data.slices[key][-1]
            self.slices[key] = torch.cat(
                (train_data.slices[key], validation_data.slices[key][1:], test_data.slices[key][1:]))

        self.processed_dataset.primary_node_labels = self.processed_dataset.x
        self.processed_dataset.node_attributes = torch.Tensor()
        self.processed_dataset.primary_edge_labels = self.processed_dataset.edge_attr
        self.processed_dataset.edge_attributes = torch.Tensor()


        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes


class QM9GraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the QM9 dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        dataset = torch_geometric.datasets.QM9(root=self.tmp_dir)
        dataset_node_labels = dataset.data.z
        dataset_node_attributes = dataset.data.x[:, [6, 7, 8, 9]]
        # one hot over the edge_attr
        dataset_edge_labels = torch.argmax(dataset.data.edge_attr, dim=1)

        dataset.data.primary_node_labels = dataset_node_labels
        dataset.data.primary_edge_labels = dataset_edge_labels
        dataset.data.node_attributes = dataset_node_attributes
        dataset.data.edge_attributes = torch.Tensor()

        self.processed_dataset = dataset.data
        self.slices = dataset.slices
        self.set_sizes()
        return self.processed_dataset, self.slices, self.sizes


class OGBGraphPropertyGraphDataPreprocessing(GraphDataPreprocessing):
    def __init__(self, name, tmp_dir="/tmp"):
        super().__init__(name, tmp_dir)
        self.preprocess()


    def preprocess(self, *args, **kwargs):
        """
        Preprocess the OGB dataset.

        :param args: Additional positional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Processed graph data.
        """
        dataset_ogb = PygGraphPropPredDataset(name=self.name, root=self.tmp_dir)
        split_idx = dataset_ogb.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        self.processed_dataset = dataset_ogb.data
        self.processed_dataset.primary_node_labels = dataset_ogb.x[:, 0]  # first column is the primary node label
        self.processed_dataset.node_attributes = dataset_ogb.x[:, 1:9]  # next 8 columns are node attributes
        self.processed_dataset.primary_edge_labels = dataset_ogb.edge_attr[:, 0]  # first column is the primary edge label
        self.processed_dataset.edge_attributes = dataset_ogb.edge_attr[:, 1:3]  # next 2 columns are edge attributes
        self.slices = dataset_ogb.slices
        self.set_sizes()

        return self.processed_dataset, self.slices, self.sizes
