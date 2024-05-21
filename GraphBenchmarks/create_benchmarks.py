from typing import List

import networkx as nx
import numpy as np
from torch_geometric.data import InMemoryDataset

from GraphBenchmarks.ExampleGraphs import Snowflakes
from GraphData.DataSplits.create_splits import create_splits
from GraphData.Distances.save_distances import save_distances
from GraphData.GraphData import get_graph_data
from GraphData.Labels.generator.save_labels import save_standard_labels, save_circle_labels, save_subgraph_labels
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


# create function description

def even_odd_rings(data_size=1200, ring_size=100, difficulty=1, count=False) -> (List[nx.Graph], List[int]):
    """
    Create a benchmark dataset consisting of labeled rings with ring_size nodes and labels.
    The label of the graph is determined by the following:
    - Select the node with label and the node with distance ring_size//2 say x and the ones with distances ring_size//4, ring_size//8, say y_1, y_2 and z_1, z_2
    Now consider the numbers:
    a = 1 + x
    b = y_1 + y_2
    c = z_1 + z_2
    and distinct the cases odd and even. This defines the 8 possible labels of the graphs.
    """
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(646843)
    class_number = 0
    permutation_storage = []
    while len(graphs) < data_size:
        G = nx.Graph()
        label_permutation = np.random.permutation(ring_size)
        for j in range(0, ring_size):
            G.add_node(j, label=label_permutation[j])
        for j in range(0, ring_size):
            G.add_edge(j % ring_size, (j + 1) % ring_size)
        # permute the Ids of the nodes
        random_permutation = np.random.permutation(ring_size)

        # make random permutation start with 0
        r_perm = np.roll(random_permutation, -np.where(random_permutation == 0)[0][0])
        # to list
        r_perm = r_perm.tolist()
        if r_perm not in permutation_storage:
            # add permutation to storage
            permutation_storage.append(r_perm)

            G = nx.relabel_nodes(G, {i: random_permutation[i] for i in range(ring_size)})
            if count:
                class_number = 2
                opposite_nodes = []
                for node in G.nodes(data=True):
                    node_label = node[1]["label"]
                    node_id = node[0]
                    pos = np.where(random_permutation == node_id)[0][0]
                    # get opposite node in the ring
                    opposite_node = random_permutation[(pos + ring_size // 2) % ring_size]
                    # get opposite node label in the ring
                    opposite_node_label = G.nodes[opposite_node]["label"]
                    # add node_label + opposite_node_label to opposite_nodes
                    opposite_nodes.append(node_label + opposite_node_label)
                # count odd and even entries in opposite_nodes
                odd_count = np.count_nonzero(np.array(opposite_nodes) % 2)
                even_count = len(opposite_nodes) - odd_count
                if odd_count > even_count:
                    label = 1
                else:
                    label = 0
            else:
                # find graph node with label 0
                for node in G.nodes(data=True):
                    if node[1]["label"] == 0:
                        node_0 = node[0]
                        break
                # get index of node_0 in random_permutation
                pos = np.where(random_permutation == node_0)[0][0]
                node_1 = random_permutation[(pos + ring_size // 4) % ring_size]
                node_2 = random_permutation[(pos + ring_size // 2) % ring_size]
                node_3 = random_permutation[(pos - ring_size // 4) % ring_size]
                # get the neighbors of node_0
                node_4 = random_permutation[(pos + 1) % ring_size]
                node_5 = random_permutation[(pos - 1 + ring_size) % ring_size]

                label_node_1 = G.nodes[node_1]["label"]
                label_node_2 = G.nodes[node_2]["label"]
                label_node_3 = G.nodes[node_3]["label"]
                label_node_4 = G.nodes[node_4]["label"]
                label_node_5 = G.nodes[node_5]["label"]

                # add the labels of the nodes
                a = 0 + label_node_2
                b = label_node_1 + label_node_3
                c = label_node_4 + label_node_5

                if difficulty == 1:
                    label = a % 2
                    class_number = 2
                elif difficulty == 2:
                    label = 2 * (a % 2) + b % 2
                    class_number = 4
                elif difficulty == 3:
                    label = 4 * (a % 2) + 2 * (b % 2) + c % 2
                    class_number = 8

            # get unique label count and append if count for label is smaller than data_size//6
            unique_labels, counts = np.unique(labels, return_counts=True)
            if label not in labels or counts[unique_labels == label] < data_size // class_number:
                graphs.append(G)
                labels.append(label)
    # shuffle the graphs and labels
    perm = np.random.permutation(len(graphs))
    graphs = [graphs[i] for i in perm]
    labels = [labels[i] for i in perm]
    return graphs, labels


def main(output_path="Data/", benchmarks=None):
    for name in benchmarks:
        if name == "EvenOddRings1_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=1, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "EvenOddRings2_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=2, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "EvenOddRings2_100":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=100, difficulty=2, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "EvenOddRings3_16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, difficulty=3, count=False)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "EvenOddRingsCount16":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=16, count=True)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "EvenOddRingsCount100":
            graphs, labels = even_odd_rings(data_size=1200, ring_size=100, count=True)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "LongRings100":
            graphs, labels = long_rings(data_size=1200, ring_size=100)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "LongRings8":
            graphs, labels = long_rings(data_size=1200, ring_size=8)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "LongRings16":
            graphs, labels = long_rings(data_size=1200, ring_size=16)
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "SnowflakesCount":
            graphs, labels = Snowflakes(smallest_snowflake=3, largest_snowflake=6, flakes_per_size=200, seed=764, generation_type="count")
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            save_circle_labels(output_path, [name], length_bound=3, cycle_type="induced", label_path="../GraphData/Labels/")
            save_circle_labels(output_path, [name], length_bound=6, cycle_type="induced", label_path="../GraphData/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(5)], id=0, label_path="../GraphData/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4)], id=1, label_path="../GraphData/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4), nx.cycle_graph(5)], id=2, label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "Snowflakes":
            graphs, labels = Snowflakes(smallest_snowflake=3, largest_snowflake=12, flakes_per_size=100, seed=764, generation_type="binary")
            save_graphs(output_path, name, graphs, labels)
            # create distance files
            save_distances(output_path, [name], cutoff=None, distance_path="../GraphData/Distances/")
            save_standard_labels(output_path, [name], label_path="../GraphData/Labels/")
            #save_circle_labels(output_path, [name], length_bound=3, cycle_type="induced", label_path="../GraphData/Labels/")
            #save_circle_labels(output_path, [name], length_bound=5, cycle_type="induced", label_path="../GraphData/Labels/")
            #save_circle_labels(output_path, [name], length_bound=6, cycle_type="induced", label_path="../GraphData/Labels/")
            #save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(5)], id=0, label_path="../GraphData/Labels/")
            #save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4)], id=1, label_path="../GraphData/Labels/")
            save_subgraph_labels(output_path, [name], subgraphs=[nx.cycle_graph(4), nx.cycle_graph(5)], id=2, label_path="../GraphData/Labels/")
            create_splits(name, path=output_path, output_path="../GraphData/DataSplits/")
        if name == "CSL":
            from GraphBenchmarks.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
            save_graphs(output_path, name, graph_data.graphs, graph_data.graph_labels)
if __name__ == "__main__":
    #main(benchmarks=["EvenOddRings1_16", "EvenOddRings2_16", "EvenOddRings3_16"])
    #main(benchmarks=["LongRings16"])
    #main(benchmarks=["LongRings100"])
    #main(benchmarks=["EvenOddRingsCount16"])
    main(benchmarks=["CSL"])
    #main(benchmarks=["LongRings100", "EvenOddRings2_16", "EvenOddRingsCount16", "Snowflakes"])
    # main(benchmarks=["EvenOddRings2_16", "EvenOddRings2_120"])
    # main(benchmarks=["EvenOddRings3_16", "EvenOddRings3_120"])

    # main(benchmarks=["LongRingsLabeled16", "LongRings100", "LongRings8"])
