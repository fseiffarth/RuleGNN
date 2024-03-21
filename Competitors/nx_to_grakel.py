from typing import List

import networkx as nx


def nx_to_grakel(nx_graphs: List[nx.Graph]):
    # create input for the kernel from the grakel graphs
    grakel_graphs = []
    for g in nx_graphs:
        edge_set = set()
        edge_dict = {}
        edges = g.edges(data=True)
        for e in edges:
            label = 0
            if 'label' in e[2]:
                label = e[2]['label']
                if len(label) == 1:
                    label = label[0]
                    try:
                        label = int(label)
                    except:
                        label = 0
            edge_dict[(e[0], e[1])] = label
            edge_dict[(e[1], e[0])] = label
            edge_set.add((e[0], e[1]))
            edge_set.add((e[1], e[0]))
        node_dict = {}
        for n in g.nodes(data=True):
            label = 0
            if 'label' in n[1]:
                label = n[1]['label']
                if len(label) == 1:
                    label = label[0]
                    try:
                        label = int(label)
                    except:
                        label = 0
            node_dict[n[0]] = label
        grakel_graphs.append([edge_set, node_dict, edge_dict])
    return grakel_graphs
