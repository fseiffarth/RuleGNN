import os
from typing import List

import networkx as nx
import numpy as np
import torch


def get_k_lowest_nonzero_indices(tensor, k):
    # Flatten the tensor
    flat_tensor = tensor.flatten()

    # Get the indices of non-zero elements
    non_zero_indices = torch.nonzero(flat_tensor, as_tuple=True)[0]

    # Select the non-zero elements
    non_zero_elements = torch.index_select(flat_tensor, 0, non_zero_indices)

    # Get the indices of the k lowest elements
    k_lowest_values, k_lowest_indices = torch.topk(non_zero_elements, k, largest=False)

    # Get the original indices
    k_lowest_original_indices = non_zero_indices[k_lowest_indices]

    return k_lowest_original_indices


def save_graphs(path, db_name, graphs: List[nx.Graph], labels: List[int] = None):
    # save in two files DBName_Nodes.txt and DBName_Edges.txt
    # DBName_Nodes.txt has the following structure GraphId NodeId Feature1 Feature2 ...
    # DBName_Edges.txt has the following structure GraphId Node1 Node2 Feature1 Feature2 ...
    # DBName_Labels.txt has the following structure GraphId Label
    # if not folder db_name exists in path create it
    if not os.path.exists(path + db_name):
        os.makedirs(path + db_name)
    # create processed and raw folders in path+db_name
    if not os.path.exists(path + db_name + "/processed"):
        os.makedirs(path + db_name + "/processed")
    if not os.path.exists(path + db_name + "/raw"):
        os.makedirs(path + db_name + "/raw")
    # update path to write into raw folder
    path = path + db_name + "/raw/"
    with open(path + db_name + "_Nodes.txt", "w") as f:
        for i, graph in enumerate(graphs):
            for node in graph.nodes(data=True):
                # get list of all data entries of the node
                data_list = list(node[1].values())
                f.write(str(i) + " " + str(node[0]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    with open(path + db_name + "_Edges.txt", "w") as f:
        for i, graph in enumerate(graphs):
            for edge in graph.edges(data=True):
                data_list = list(edge[2].values())
                f.write(str(i) + " " + str(edge[0]) + " " + str(edge[1]) + " " + " ".join(map(str, data_list)) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()
    with open(path + db_name + "_Labels.txt", "w") as f:
        if labels is not None:
            for i, label in enumerate(labels):
                if type(label) == int:
                    f.write(str(i) + " " + str(label) + "\n")
                elif type(label) == np.ndarray or type(label) == list:
                    f.write(str(i) + " " + " ".join(map(str, label)) + "\n")
        else:
            for i in range(len(graphs)):
                f.write(str(i) + " " + str(0) + "\n")
        # remove last empty line
        f.seek(f.tell() - 1, 0)
        f.truncate()


def load_graphs(path, db_name):
    graphs = []
    labels = []
    with open(path + db_name + "_Nodes.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node_id = int(data[1])
            feature = list(map(float, data[2:]))
            while len(graphs) <= graph_id:
                graphs.append(nx.Graph())
            graphs[graph_id].add_node(node_id, label=feature)
    with open(path + db_name + "_Edges.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            node1 = int(data[1])
            node2 = int(data[2])
            feature = list(map(float, data[3:]))
            graphs[graph_id].add_edge(node1, node2, label=feature)
    with open(path + db_name + "_Labels.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            data = line.strip().split(" ")
            graph_id = int(data[0])
            if len(data) == 2:
                label = int(data[1])
            else:
                label = list(map(float, data[1:]))

            while len(labels) <= graph_id:
                labels.append(label)
            labels[graph_id] = label
    return graphs, labels
