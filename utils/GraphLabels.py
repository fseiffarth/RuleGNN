from collections import OrderedDict
from typing import List

from GraphData.Labels.generator.save_labels import save_node_labels


class NodeLabels:
    def __init__(self, node_labels=None):
        self.node_labels = None
        self.unique_node_labels = None
        self.db_unique_node_labels = None
        self.num_unique_node_labels = 0

        if node_labels is not None:
            self.node_labels = node_labels
            self.db_unique_node_labels = {}
            self.unique_node_labels = []
            for g_labels in node_labels:
                self.unique_node_labels.append({})
                for l in g_labels:
                    if l not in self.db_unique_node_labels:
                        self.db_unique_node_labels[l] = 1
                    else:
                        self.db_unique_node_labels[l] += 1
                    if l not in self.unique_node_labels[-1]:
                        self.unique_node_labels[-1][l] = 1
                    else:
                        self.unique_node_labels[-1][l] += 1
            self.num_unique_node_labels = len(self.db_unique_node_labels)



def combine_node_labels(labels: List[NodeLabels]):
    # create tuples for each node
    node_labels = []
    label_map = {}
    for i, g_labels in enumerate(labels[0].node_labels):
        node_labels.append([])
        for j, l in enumerate(g_labels):
            label_tuple = []
            for k in range(len(labels)):
                label_tuple.append(labels[k].node_labels[i][j])
            if tuple(label_tuple) not in label_map:
                label_map[tuple(label_tuple)] = 1
            else:
                label_map[tuple(label_tuple)] += 1
            node_labels[-1].append(tuple(label_tuple))

    # sort dict by values
    label_map = OrderedDict(sorted(label_map.items(), key=lambda item: item[1], reverse=True))

    index_map = {}
    # iterate over ordered dict and create map
    for i, key in enumerate(label_map):
        index_map[key] = i


    # create new labels from node labels using the map
    new_labels = []
    for g_labels in node_labels:
        new_labels.append([])
        for l in g_labels:
            new_labels[-1].append(index_map[l])

    return NodeLabels(new_labels)

class EdgeLabels:
    def __init__(self):
        self.edge_labels = None
        self.unique_edge_labels = None
        self.db_unique_edge_labels = None
        self.num_unique_edge_labels = 0