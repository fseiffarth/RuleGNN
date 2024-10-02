#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

import json
import os
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from networkx import normalized_laplacian_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import RandomSampler
from torch_geometric.data import DataLoader

class GraphDataset:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def get_targets(self):
        targets = [d.y.item() for d in self.data]
        return np.array(targets)

    def get_data(self):
        return self.data

    def augment(self, v_outs=None, e_outs=None, g_outs=None, o_outs=None):
        """
        v_outs must have shape |G|x|V_g| x L x ? x ...
        e_outs must have shape |G|x|E_g| x L x ? x ...
        g_outs must have shape |G| x L x ? x ...
        o_outs has arbitrary shape, it is a handle for saving extra things
        where    L = |prev_outputs_to_consider|.
        The graph order in which these are saved i.e. first axis, should reflect the ones in which
        they are saved in the original dataset.
        :param v_outs:
        :param e_outs:
        :param g_outs:
        :param o_outs:
        :return:
        """
        for index in range(len(self)):
            if v_outs is not None:
                self[index].v_outs = v_outs[index]
            if e_outs is not None:
                self[index].e_outs = e_outs[index]
            if g_outs is not None:
                self[index].g_outs = g_outs[index]
            if o_outs is not None:
                self[index].o_outs = o_outs[index]


class GraphDatasetSubset(GraphDataset):
    """
    Subsets the dataset according to a list of indices.
    """

    def __init__(self, data, indices):
        super().__init__(data)
        self.indices = indices

    def __getitem__(self, index):
        return self.data[self.indices[index]]

    def __len__(self):
        return len(self.indices)

    def get_targets(self):
        targets = [self.data[i].y.item() for i in self.indices]
        return np.array(targets)


class GraphDatasetManager:
    def __init__(self, kfold_class=StratifiedKFold, outer_k=10, inner_k=None, seed=42, holdout_test_size=0.1,
                 use_node_degree=False, use_node_attrs=False, use_one=False, precompute_kron_indices=False,
                 max_reductions=10, DATA_DIR='DATA'):

        self.root_dir = Path(DATA_DIR) / self.name
        self.kfold_class = kfold_class
        self.holdout_test_size = holdout_test_size
        self.use_node_degree = use_node_degree
        self.use_node_attrs = use_node_attrs
        self.use_one = use_one
        self.precompute_kron_indices = precompute_kron_indices
        self.KRON_REDUCTIONS = max_reductions  # will compute indices for 10 pooling layers --> approximately 1000 nodes

        self.outer_k = outer_k
        assert (outer_k is not None and outer_k > 0) or outer_k is None

        self.inner_k = inner_k
        assert (inner_k is not None and inner_k > 0) or inner_k is None

        self.seed = seed

        self.raw_dir = self.root_dir / "raw"
        if not self.raw_dir.exists():
            os.makedirs(self.raw_dir)
            self._download()

        self.processed_dir = self.root_dir / "processed"
        if not (self.processed_dir / f"{self.name}.pt").exists():
            if not self.processed_dir.exists():
                os.makedirs(self.processed_dir)
            self._process()

        self.dataset = GraphDataset(torch.load(
            self.processed_dir / f"{self.name}.pt", weights_only=True))

        splits_filename = self.processed_dir / f"{self.name}_splits.json"
        if not splits_filename.exists():
            self.splits = []
            self._make_splits()
        else:
            self.splits = json.load(open(splits_filename, "r"))

    @property
    def num_graphs(self):
        return len(self.dataset)

    @property
    def dim_target(self):
        if not hasattr(self, "_dim_target") or self._dim_target is None:
            # not very efficient, but it works
            # todo not general enough, we may just remove it
            self._dim_target = np.unique(self.dataset.get_targets()).size
        return self._dim_target

    @property
    def dim_features(self):
        if not hasattr(self, "_dim_features") or self._dim_features is None:
            # not very elegant, but it works
            # todo not general enough, we may just remove it
            self._dim_features = self.dataset.data[0].x.size(1)
        return self._dim_features

    def _process(self):
        raise NotImplementedError

    def _download(self):
        raise NotImplementedError

    def _make_splits(self):
        """
        DISCLAIMER: train_test_split returns a SUBSET of the input indexes,
            whereas StratifiedKFold.split returns the indexes of the k subsets, starting from 0 to ...!
        """

        targets = self.dataset.get_targets()
        all_idxs = np.arange(len(targets))

        if self.outer_k is None:  # holdout assessment strategy
            assert self.holdout_test_size is not None

            if self.holdout_test_size == 0:
                train_o_split, test_split = all_idxs, []
            else:
                outer_split = train_test_split(all_idxs,
                                               stratify=targets,
                                               test_size=self.holdout_test_size)
                train_o_split, test_split = outer_split
            split = {"test": all_idxs[test_split], 'model_selection': []}

            train_o_targets = targets[train_o_split]

            if self.inner_k is None:  # holdout model selection strategy
                if self.holdout_test_size == 0:
                    train_i_split, val_i_split = train_o_split, []
                else:
                    train_i_split, val_i_split = train_test_split(train_o_split,
                                                                  stratify=train_o_targets,
                                                                  test_size=self.holdout_test_size)
                split['model_selection'].append(
                    {"train": train_i_split, "validation": val_i_split})

            else:  # cross validation model selection strategy
                inner_kfold = self.kfold_class(
                    n_splits=self.inner_k, shuffle=True)
                for train_ik_split, val_ik_split in inner_kfold.split(train_o_split, train_o_targets):
                    split['model_selection'].append(
                        {"train": train_o_split[train_ik_split], "validation": train_o_split[val_ik_split]})

            self.splits.append(split)

        else:  # cross validation assessment strategy

            outer_kfold = self.kfold_class(
                n_splits=self.outer_k, shuffle=True)

            for train_ok_split, test_ok_split in outer_kfold.split(X=all_idxs, y=targets):
                split = {"test": all_idxs[test_ok_split], 'model_selection': []}

                train_ok_targets = targets[train_ok_split]

                if self.inner_k is None:  # holdout model selection strategy
                    assert self.holdout_test_size is not None
                    train_i_split, val_i_split = train_test_split(train_ok_split,
                                                                  stratify=train_ok_targets,
                                                                  test_size=self.holdout_test_size)
                    split['model_selection'].append(
                        {"train": train_i_split, "validation": val_i_split})

                else:  # cross validation model selection strategy
                    inner_kfold = self.kfold_class(
                        n_splits=self.inner_k, shuffle=True)
                    for train_ik_split, val_ik_split in inner_kfold.split(train_ok_split, train_ok_targets):
                        split['model_selection'].append(
                            {"train": train_ok_split[train_ik_split], "validation": train_ok_split[val_ik_split]})

                self.splits.append(split)

        filename = self.processed_dir / f"{self.name}_splits.json"
        with open(filename, "w") as f:
            json.dump(self.splits[:], f, cls=NumpyEncoder)

    def _get_loader(self, dataset, batch_size=1, shuffle=True):
        # dataset = GraphDataset(data)
        sampler = RandomSampler(dataset) if shuffle is True else None

        # 'shuffle' needs to be set to False when instantiating the DataLoader,
        # because pytorch  does not allow to use a custom sampler with shuffle=True.
        # Since our shuffler is a random shuffler, either one wants to do shuffling
        # (in which case he should instantiate the sampler and set shuffle=False in the
        # DataLoader) or he does not (in which case he should set sampler=None
        # and shuffle=False when instantiating the DataLoader)

        return DataLoader(dataset,
                          batch_size=batch_size,
                          sampler=sampler,
                          shuffle=False,  # if shuffle is not None, must stay false, ow is shuffle is false
                          pin_memory=True)

    def get_test_fold(self, outer_idx, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0

        idxs = self.splits[outer_idx]["test"]
        test_data = GraphDatasetSubset(self.dataset.get_data(), idxs)

        if len(test_data) == 0:
            test_loader = None
        else:
            test_loader = self._get_loader(test_data, batch_size, shuffle)

        return test_loader

    def get_model_selection_fold(self, outer_idx, inner_idx=None, batch_size=1, shuffle=True):
        outer_idx = outer_idx or 0
        inner_idx = inner_idx or 0

        idxs = self.splits[outer_idx]["model_selection"][inner_idx]
        train_data = GraphDatasetSubset(self.dataset.get_data(), idxs["train"])
        val_data = GraphDatasetSubset(self.dataset.get_data(), idxs["validation"])

        train_loader = self._get_loader(train_data, batch_size, shuffle)

        if len(val_data) == 0:
            val_loader = None
        else:
            val_loader = self._get_loader(val_data, batch_size, shuffle)

        return train_loader, val_loader


class BenchmarkDatasetManager(GraphDatasetManager):
    classification = True

    def _process(self):
        # load from raw path
        graphs, labels = self.load_graphs(self.raw_dir, self.name)

        # create graphs_data a dict with keys:
        # graph_nodes, graph_edges, node_labels, node_attrs, edge_labels, edge_attrs
        graphs_data = {'graph_nodes': defaultdict(list), 'graph_edges': defaultdict(list),
                       'node_labels': defaultdict(list), 'node_attrs': defaultdict(list),
                       'edge_labels': defaultdict(list), 'edge_attrs': defaultdict(list)}
        id_counter = 1
        node_label_set = set()
        edge_label_set = set()
        for i, graph in enumerate(graphs, 1):
            node_ids = [i + id_counter for i in range(graph.number_of_nodes())]
            edge_ids = [[i + id_counter, j + id_counter] for i, j in graph.edges()]
            node_labels = [0] * graph.number_of_nodes()
            node_attrs = [None] * graph.number_of_nodes()
            # get attr data from nodes
            attrs_data = nx.get_node_attributes(graph, 'attr')
            numpy_attrs_data = None
            # if attrs_data is not empty, then convert to numpy array
            if attrs_data:
                numpy_attrs_data = np.array(list(attrs_data.values())).T
                # normalize the by dividing by the maximum value per row
                numpy_attrs_data = numpy_attrs_data / numpy_attrs_data.max(axis=1)[:, None]
                # transpose the matrix back
                numpy_attrs_data = numpy_attrs_data.T
            for node in graph.nodes(data=True):
                if type(node[1]['label']) == list:
                    node_label = int(node[1]['label'][0])
                elif type(node[1]['label']) == int:
                    node_label = node[1]['label']
                node_label_set.add(node_label)
                node_labels[node[0]] = node_label
                if numpy_attrs_data is not None:
                    node_attrs[node[0]] = numpy_attrs_data[node[0]]
            graphs_data['graph_nodes'][i] = node_ids
            graphs_data['graph_edges'][i] = edge_ids
            graphs_data['node_labels'][i] = node_labels
            # check wheter there is at least entry in node_attrs that is not None
            if node_attrs[0] is not None:
                graphs_data['node_attrs'][i] = node_attrs
            id_counter += graph.number_of_nodes()
        # targets is a list of the labels (integer)
        targets = labels

        num_edge_labels = len(edge_label_set)
        num_node_labels = len(node_label_set)

        max_num_nodes = max([len(v) for (k, v) in graphs_data['graph_nodes'].items()])
        setattr(self, 'max_num_nodes', max_num_nodes)

        dataset = []
        for i, target in enumerate(targets, 1):
            graph_data = {k: v[i] for (k, v) in graphs_data.items()}
            G = create_graph_from_tu_data(graph_data, target, num_node_labels, num_edge_labels)

            if self.precompute_kron_indices:
                laplacians, v_plus_list = self._precompute_kron_indices(G)
                G.laplacians = laplacians
                G.v_plus = v_plus_list

            if G.number_of_nodes() > 1 and G.number_of_edges() > 0:
                data = self._to_data(G)
                dataset.append(data)

        torch.save(dataset, self.processed_dir / f"{self.name}.pt")

    def _to_data(self, G):
        datadict = {}

        node_features = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        datadict.update(x=node_features)

        if G.laplacians is not None:
            datadict.update(laplacians=G.laplacians)
            datadict.update(v_plus=G.v_plus)

        edge_index = G.get_edge_index()
        datadict.update(edge_index=edge_index)

        if G.has_edge_attrs:
            edge_attr = G.get_edge_attr()
            datadict.update(edge_attr=edge_attr)

        target = G.get_target(classification=self.classification)
        datadict.update(y=target)

        data = Data(**datadict)

        return data

    def _precompute_kron_indices(self, G):
        laplacians = []  # laplacian matrices (represented as 1D vectors)
        v_plus_list = []  # reduction matrices

        X = G.get_x(self.use_node_attrs, self.use_node_degree, self.use_one)
        lap = torch.Tensor(normalized_laplacian_matrix(G).todense())  # I - D^{-1/2}AD^{-1/2}
        # print(X.shape, lap.shape)

        laplacians.append(lap)

        for _ in range(self.KRON_REDUCTIONS):
            if lap.shape[0] == 1:  # Can't reduce further:
                v_plus, lap = torch.tensor([1]), torch.eye(1)
                # print(lap.shape)
            else:
                v_plus, lap = self._vertex_decimation(lap)
                # print(lap.shape)
                # print(lap)

            laplacians.append(lap.clone())
            v_plus_list.append(v_plus.clone().long())

        return laplacians, v_plus_list

    # For the Perron–Frobenius theorem, if A is > 0 for all ij then the leading eigenvector is > 0
    # A Laplacian matrix is symmetric (=> diagonalizable)
    # and dominant eigenvalue (true in most cases? can we enforce it?)
    # => we have sufficient conditions for power method to converge
    def _power_iteration(self, A, num_simulations=30):
        # Ideally choose a random vector
        # To decrease the chance that our vector
        # Is orthogonal to the eigenvector
        b_k = torch.rand(A.shape[1]).unsqueeze(dim=1) * 0.5 - 1

        for _ in range(num_simulations):
            # calculate the matrix-by-vector product Ab
            b_k1 = torch.mm(A, b_k)

            # calculate the norm
            b_k1_norm = torch.norm(b_k1)

            # re normalize the vector
            b_k = b_k1 / b_k1_norm

        return b_k

    def _vertex_decimation(self, L):

        max_eigenvec = self._power_iteration(L)
        v_plus, v_minus = (max_eigenvec >= 0).squeeze(), (max_eigenvec < 0).squeeze()

        # print(v_plus, v_minus)

        # diagonal matrix, swap v_minus with v_plus not to incur in errors (does not change the matrix)
        if torch.sum(v_plus) == 0.:  # The matrix is diagonal, cannot reduce further
            if torch.sum(v_minus) == 0.:
                assert v_minus.shape[0] == L.shape[0], (v_minus.shape, L.shape)
                # I assumed v_minus should have ones, but this is not necessarily the case. So I added this if
                return torch.ones(v_minus.shape), L
            else:
                return v_minus, L

        L_plus_plus = L[v_plus][:, v_plus]
        L_plus_minus = L[v_plus][:, v_minus]
        L_minus_minus = L[v_minus][:, v_minus]
        L_minus_plus = L[v_minus][:, v_plus]

        L_new = L_plus_plus - torch.mm(torch.mm(L_plus_minus, torch.inverse(L_minus_minus)), L_minus_plus)

        return v_plus, L_new

    def _precompute_assignments(self):
        pass

    @staticmethod
    def load_graphs(path, db_name):
        graphs = []
        labels = []
        # PosixPath to string
        path = str(path)
        with open(path + "/" + db_name + "_Nodes.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                graph_id = int(data[0])
                node_id = int(data[1])
                label = int(data[2])
                # the rest of the line is the node attributes#
                node_attrs = list(map(float, data[3:]))
                while len(graphs) <= graph_id:
                    graphs.append(nx.Graph())
                graphs[graph_id].add_node(node_id, label=label, attr=node_attrs)
        with open(path + "/" + db_name + "_Edges.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                graph_id = int(data[0])
                node1 = int(data[1])
                node2 = int(data[2])
                if len(data) > 3:
                    label = int(data[3])
                    if len(data) > 4:
                        # the rest of the line is the edge attributes
                        edge_attrs = list(map(float, data[4:]))
                        graphs[graph_id].add_edge(node1, node2, label=label, attr=edge_attrs)
                    else:
                        graphs[graph_id].add_edge(node1, node2, label=label)
                else:
                    graphs[graph_id].add_edge(node1, node2)
        with open(path + "/" + db_name + "_Labels.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                data = line.strip().split(" ")
                graph_id = int(data[0])
                label = int(data[1])
                while len(labels) <= graph_id:
                    labels.append(label)
                labels[graph_id] = label
        return graphs, labels
