from typing import List

import networkx as nx
import numpy as np

from src.utils.BenchmarkDatasetGeneration.RingTransfer import RingTransfer
from src.utils.snowflake_generation import Snowflakes


def long_rings(data_size=1200, ring_size=100, seed=764) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    # seed numpy
    np.random.seed(seed)
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

def even_odd_rings(data_size=1200, ring_size=100, difficulty=1, count=False, seed=764) -> (List[nx.Graph], List[int]):
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
    np.random.seed(seed)
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

def ring_diagonals( data_size=1200, ring_size=100) -> (List[nx.Graph], List[int]):
    """
    Create a dataset of ring graphs with diagonals.
    :param data_size: number of graphs to create
    :param ring_size: number of nodes in each ring
    :return: a list of graphs and a list of labels
    """
    graphs = []
    labels = []
    seed = 16
    np.random.seed(seed)
    class_counter = [0, 0]
    while sum(class_counter) < data_size:
        G = nx.cycle_graph(ring_size)
        # add random 1-dim labels and 3-dim features to nodes and edges
        for node in G.nodes():
            G.nodes[node]['label'] = np.random.randint(0, 2)
            G.nodes[node]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]
        for edge in G.edges():
            G[edge[0]][edge[1]]['label'] = np.random.randint(0, 2)
            G[edge[0]][edge[1]]['feature'] = [np.random.rand(), np.random.rand(), np.random.rand()]

        # get two random nodes in the ring and connect them with an edge
        diag_start = np.random.randint(ring_size)
        while True:
            diag_end = np.random.randint(ring_size)
            if diag_end != diag_start:
                break
        # get the distance in the ring between the two nodes
        dist = nx.shortest_path_length(G, diag_start, diag_end)
        G.add_edge(diag_start, diag_end)
        # determine the label of the graph G
        # Case 1: Edge Label of the diagonal is 1
        # Case 2: Labels of the two end nodes of the diagonal are the same
        # Case 3: Distance between the two end nodes of the diagonal greater than 25
        # => Then the graph label is 1, else 0
        graph_label = 0
        edge = G.edges[diag_start, diag_end]
        if 'label' in edge and edge['label'] == 1:
            graph_label = 1
        elif G.nodes[diag_start]['label'] == G.nodes[diag_end]['label']:
            graph_label = 1
        elif dist > 13:
            graph_label = 1
        if class_counter[graph_label] >= data_size / 2:
            continue
        else:
            class_counter[graph_label] += 1
            labels.append(graph_label)
            graphs.append(G)

    return graphs, labels

def snowflakes(smallest_snowflake=3, largest_snowflake=12, flakes_per_size=100, seed=764, generation_type='binary') -> (List[nx.Graph], List[int]):
    """
    Create a dataset of snowflake graphs.
    """
    return Snowflakes(smallest_snowflake=smallest_snowflake, largest_snowflake=largest_snowflake, flakes_per_size=flakes_per_size, plot=False, seed=seed, generation_type=generation_type)

def csl_graphs() -> (List[nx.Graph], List[int]):
    from src.Preprocessing.csl import CSL
    csl = CSL()
    graph_data = csl.get_graphs(with_distances=False)
    return graph_data.graphs, graph_data.graph_labels

def ring_transfer(data_size=1200, node_dimension=10, ring_size=100, seed=764) -> (List[nx.Graph], List[np.ndarray[float]]):
    return RingTransfer(data_size=data_size, node_dimension=node_dimension, ring_size=ring_size, seed=seed)

def parity_check(data_size=1500, max_size=40, seed=764) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, label=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]["label"] = rand_sequence[j]
        graphs.append(G)
        # count number of 1s in the sequence
        even = np.count_nonzero(rand_sequence) % 2
        labels.append(even)
    return graphs, labels

def even_pairs(data_size=1500, max_size=40, seed=764) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, label=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]["label"] = rand_sequence[j]
        graphs.append(G)
        # check wheter first and last node have the same label 0
        valid = rand_sequence[0] == rand_sequence[-1]
        labels.append(valid)
    return graphs, labels

def first_a(data_size=1500, max_size=40, seed=764) -> (List[nx.Graph], List[int]):
    graphs = []
    labels = []
    np.random.seed(seed)
    for i in range(data_size):
        size = np.random.randint(1, max_size + 1)
        G = nx.Graph()
        for j in range(size):
            G.add_node(j, label=0)
        for j in range(size - 1):
            G.add_edge(j, j + 1)
        # create random seqeunce of size size of 0s and 1s
        rand_sequence = np.random.randint(0, 2, size)
        # assign the labels to the nodes
        for j in range(size):
            G.nodes[j]["label"] = rand_sequence[j]
        graphs.append(G)
        # check wheter first and last node have the same label 0
        valid = rand_sequence[0]
        labels.append(valid)
    return graphs, labels