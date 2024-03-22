# generate WL labels for the graph data and save them to a file
import os

import networkx as nx
import numpy as np

from GraphData import GraphData, NodeLabeling

def write_node_labels(file, node_labels):
    with open(file, 'w') as f:
        for i, g_labels in enumerate(node_labels):
            for j, l in enumerate(g_labels):
                if j < len(g_labels) - 1:
                    f.write(f"{l} ")
                else:
                    if i != len(node_labels) - 1:
                        f.write(f"{l}\n")
                    else:
                        f.write(f"{l}")

def save_wl_labels(data_path):
    for db_name in ['IMDB-BINARY', 'IMDB-MULTI', 'DD', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']:
        # load the graph data'NCI1', 'NCI109', 'Mutagenicity', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        graph_data = GraphData.GraphData()
        graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                      relabel_nodes=True, use_features=False, use_attributes=False)

        node_labels = graph_data.node_labels['primary'].node_labels
        # save the node labels to a file
        # save node_labels as numpy array
        file = f"{db_name}_primary_labels.txt"
        write_node_labels(file, node_labels)


        graph_data.add_node_labels(node_labeling_name='wl_0', max_label_num=-1,
                                   node_labeling_method=NodeLabeling.degree_node_labeling)
        node_labels = graph_data.node_labels['wl_0'].node_labels
        # save the node labels to a file
        # save node_labels as numpy array
        file = f"{db_name}_wl_0_labels.txt"
        write_node_labels(file, node_labels)

        for l in ['wl_1', 'wl_2']:
            iterations = int(l.split("_")[1])
            for n_node_labels in [100, 500, 50000]:
                if iterations > 0:
                    graph_data.add_node_labels(node_labeling_name=l, max_label_num=n_node_labels,
                                               node_labeling_method=NodeLabeling.weisfeiler_lehman_node_labeling,
                                               max_iterations=iterations)
                node_labels = graph_data.node_labels[l].node_labels
                # save the node labels to a file
                # save node_labels as numpy array
                file = f"{db_name}_{l}_{n_node_labels}_labels.txt"
                write_node_labels(file, node_labels)


def save_circle_labels(data_path):
    for db_name in ['MUTAG']:
        # load the graph data'NCI1', 'NCI109', 'Mutagenicity', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        graph_data = GraphData.GraphData()
        graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                      relabel_nodes=True, use_features=False, use_attributes=False)
        cycle_dict = []
        for graph in graph_data.graphs:
            cycle_dict.append({})
            cycles = nx.chordless_cycles(graph, 6)
            for cycle in cycles:
                for node in cycle:
                    if node in cycle_dict[-1]:
                        if len(cycle) in cycle_dict[-1][node]:
                            cycle_dict[-1][node][len(cycle)] += 1
                        else:
                            cycle_dict[-1][node][len(cycle)] = 1
                    else:
                        cycle_dict[-1][node] = {}
                        cycle_dict[-1][node][len(cycle)] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in cycle_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i): i for i in dict_list}.values())

        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.graphs):
            labels.append([])
            for node in graph.nodes():
                if node in cycle_dict[graph_id]:
                    TODO continue here


        file = f"{db_name}_primary_labels.txt"
        write_node_labels(file, node_labels)
        return cycles



def main():
    data_path = "../../../GraphData/DS_all/"
    #save_wl_labels(data_path)
    save_circle_labels(data_path)




if __name__ == '__main__':
    main()
