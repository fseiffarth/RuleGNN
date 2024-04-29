from typing import List

import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset

from GraphData.DataSplits.create_splits import create_splits
from GraphData.Distances.save_distances import save_distances
from GraphData.Labels.generator.save_labels import save_standard_labels
from utils.utils import save_graphs

def long_rings(data_size=1200, ring_size=100) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(646843)
    while len(graphs) < data_size:
        G = nx.Graph()
        for j in range(0, ring_size):
            G.add_node(j, label=0)
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)
        G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
        # get a random node and the one on the opposite and the one on 90 degree and 270 and assign random labels from the list {1,2,3,4}
        pos = np.random.randint(0, ring_size)
        node_0 = random_permutation[pos]
        node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
        node_2 = random_permutation[(pos + ring_size // 2) % ring_size]
        node_3 = random_permutation[(pos + 3 * ring_size // 4) % ring_size]
        # randomly shuffle {1,2,3,4} and assign to the nodes
        rand_perm = np.random.permutation([1, 2, 3, 4])
        # change the labels of the nodes
        G.nodes[node_0]["label"] = rand_perm[0]
        G.nodes[node_1]["label"] = rand_perm[1]
        G.nodes[node_2]["label"] = rand_perm[2]
        G.nodes[node_3]["label"] = rand_perm[3]
        # find position of 1 in rand_perm
        pos_one = np.where(rand_perm == 1)[0][0]
        # find label opposite to 1
        pos_opp = (pos_one + 2) % 4
        label_opp = rand_perm[pos_opp]
        if label_opp == 2:
            label = 0
        elif label_opp == 3:
            label = 1
        elif label_opp == 4:
            label = 2
        # get unique label count and append if count for label is smaller than data_size//6
        unique_labels, counts = np.unique(labels, return_counts=True)
        if label not in labels or counts[unique_labels == label] < data_size // 3:
            graphs.append(G)
            labels.append(label)
    # shuffle the graphs and labels
    perm = np.random.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    return graphs, labels


def main(output_path="Data/", benchmark=None):
    if "LongRings" in benchmark:
        graphs, labels = long_rings(data_size=1200, ring_size=100)
        save_graphs(output_path, "LongRings", graphs, labels)
        # create distance files
        save_distances(output_path, ["LongRings"], cutoff=None, distance_path="../GraphData/Distances/")
        save_standard_labels(output_path, ["LongRings"], label_path="../GraphData/Labels/")
        create_splits("LongRings", path=output_path, output_path="../GraphData/DataSplits/")

    pass


if __name__ == "__main__":
    main(benchmarks=["LongRings"])
