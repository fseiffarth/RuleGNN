# generate WL labels for the graph data and save them to a file
import os
from typing import List

import networkx as nx
import numpy as np
from networkx.algorithms import isomorphism
from networkx.algorithms.isomorphism import GraphMatcher
from torch_geometric.datasets import ZINC

from GraphData import GraphData, NodeLabeling
from GraphData.GraphData import zinc_to_graph_data


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


def save_wl_labels(data_path, db_names):
    for db_name in db_names:
        # load the graph data'NCI1', 'NCI109', 'NCI109', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        if db_name == 'CSL':
            from LoadData.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        elif db_name == 'ZINC':
            zinc_train = ZINC(root="../../ZINC/", subset=True, split='train')
            zinc_val = ZINC(root="../../ZINC/", subset=True, split='val')
            zinc_test = ZINC(root="../../ZINC/", subset=True, split='test')
            graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        else:
            graph_data = GraphData.GraphData()
            graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                          relabel_nodes=True, use_features=False, use_attributes=False)
        node_labels = graph_data.node_labels['primary'].node_labels
        # save the node labels to a file
        # save node_labels as numpy array
        file = f"../{db_name}_primary_labels.txt"
        write_node_labels(file, node_labels)

        graph_data.add_node_labels(node_labeling_name='wl_0', max_label_num=-1,
                                   node_labeling_method=NodeLabeling.degree_node_labeling)
        node_labels = graph_data.node_labels['wl_0'].node_labels
        # save the node labels to a file
        # save node_labels as numpy array
        file = f"../{db_name}_wl_0_labels.txt"
        write_node_labels(file, node_labels)

        for l in ['wl_1', 'wl_2']:
            iterations = int(l.split("_")[1])
            for n_node_labels in [100, 500, 50000]:
                if iterations > 0:
                    graph_data.add_node_labels(node_labeling_name=l, max_label_num=n_node_labels,
                                               node_labeling_method=NodeLabeling.weisfeiler_lehman_node_labeling,
                                               max_iterations=iterations)
                node_labels = graph_data.node_labels[f'{l}_{n_node_labels}'].node_labels
                # save the node labels to a file
                # save node_labels as numpy array
                file = f"../{db_name}_{l}_{n_node_labels}_labels.txt"
                write_node_labels(file, node_labels)


def save_circle_labels(data_path, db_names, length_bound=6, max_node_labels=None, cycle_type='simple'):
    for db_name in db_names:
        # load the graph data'NCI1', 'NCI109', 'NCI109', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        if db_name == 'CSL':
            from LoadData.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        elif db_name == 'ZINC':
            zinc_train = ZINC(root="../../ZINC/", subset=True, split='train')
            zinc_val = ZINC(root="../../ZINC/", subset=True, split='val')
            zinc_test = ZINC(root="../../ZINC/", subset=True, split='test')
            graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        else:
            graph_data = GraphData.GraphData()
            graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                          relabel_nodes=True, use_features=False, use_attributes=False)
        cycle_dict = []
        for graph in graph_data.graphs:
            cycle_dict.append({})
            if cycle_type == 'simple':
                cycles = nx.simple_cycles(graph, length_bound)
            elif cycle_type == 'induced':
                cycles = nx.chordless_cycles(graph, length_bound)
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

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}


        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.graphs):
            labels.append([])
            for node in graph.nodes():
                if node in cycle_dict[graph_id]:
                    cycle_d = str(cycle_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))
        max_node_labels_str = ''
        if max_node_labels is not None:
            max_node_labels_str = f"_{max_node_labels}"
            relabel_most_frequent_node_labels(labels, max_node_labels)
        if cycle_type == 'simple':
            file = f"../{db_name}_simple_cycles_{length_bound}{max_node_labels_str}_labels.txt"
        elif cycle_type == 'induced':
            file = f"../{db_name}_induced_cycles_{length_bound}{max_node_labels_str}_labels.txt"
        write_node_labels(file, labels)


def save_subgraph_labels(data_path, db_names, subgraphs=List[nx.Graph]):
    for db_name in db_names:
        # load the graph data'NCI1', 'NCI109', 'NCI109', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        if db_name == 'CSL':
            from LoadData.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        elif db_name == 'ZINC':
            zinc_train = ZINC(root="../../ZINC/", subset=True, split='train')
            zinc_val = ZINC(root="../../ZINC/", subset=True, split='val')
            zinc_test = ZINC(root="../../ZINC/", subset=True, split='test')
            graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        else:
            graph_data = GraphData.GraphData()
            graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                          relabel_nodes=True, use_features=False, use_attributes=False)
        subgraph_dict = []
        for graph in graph_data.graphs:
            subgraph_dict.append({})
            for i, subgraph in enumerate(subgraphs):
                GM = GraphMatcher(graph, subgraph)
                for x in GM.subgraph_isomorphisms_iter():
                    for node in x:
                        if node in subgraph_dict[-1]:
                            if i in subgraph_dict[-1][node]:
                                subgraph_dict[-1][node][i] += 1
                            else:
                                subgraph_dict[-1][node][i] = 1
                        else:
                            subgraph_dict[-1][node] = {}
                            subgraph_dict[-1][node][i] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in subgraph_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}


        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.graphs):
            labels.append([])
            for node in graph.nodes():
                if node in subgraph_dict[graph_id]:
                    cycle_d = str(subgraph_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))
        file = f"../{db_name}_subgraphs1_labels.txt"
        write_node_labels(file, labels)



def save_clique_labels(data_path, db_names, max_clique=6):
    for db_name in db_names:
        # load the graph data'NCI1', 'NCI109', 'NCI109', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
        if db_name == 'CSL':
            from LoadData.csl import CSL
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
        elif db_name == 'ZINC':
            zinc_train = ZINC(root="../../ZINC/", subset=True, split='train')
            zinc_val = ZINC(root="../../ZINC/", subset=True, split='val')
            zinc_test = ZINC(root="../../ZINC/", subset=True, split='test')
            graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
        else:
            graph_data = GraphData.GraphData()
            graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                          relabel_nodes=True, use_features=False, use_attributes=False)
        clique_dict = []
        for graph in graph_data.graphs:
            clique_dict.append({})
            cliques = list(nx.find_cliques(graph))
            for clique in cliques:
                if len(clique) <= max_clique:
                    for node in clique:
                        if node in clique_dict[-1]:
                            if len(clique) in clique_dict[-1][node]:
                                clique_dict[-1][node][len(clique)] += 1
                            else:
                                clique_dict[-1][node][len(clique)] = 1
                        else:
                            clique_dict[-1][node] = {}
                            clique_dict[-1][node][len(clique)] = 1

        # get all unique dicts of cycles
        dict_list = []
        for g in clique_dict:
            for node_id, c_dict in g.items():
                dict_list.append(c_dict)

        dict_list = list({str(i) for i in dict_list})
        # sort the dict_list
        dict_list = sorted(dict_list)
        label_dict = {key: value for key, value in zip(dict_list, range(len(dict_list)))}


        # set the node labels
        labels = []
        for graph_id, graph in enumerate(graph_data.graphs):
            labels.append([])
            for node in graph.nodes():
                if node in clique_dict[graph_id]:
                    cycle_d = str(clique_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))

        file = f"../{db_name}_cliques_{max_clique}_labels.txt"
        write_node_labels(file, labels)

def relabel_most_frequent_node_labels(node_labels, max_node_labels):
    """
    Relabel the node labels with the most frequent labels.
    :param node_labels: list of lists
    :param max_node_labels: int
    :return: list of lists
    """
    # get the unique node labels toghether with their frequency
    unique_frequency = {}
    for g_labels in node_labels:
        for l in g_labels:
            if l in unique_frequency:
                unique_frequency[l] += 1
            else:
                unique_frequency[l] = 1
    if len(unique_frequency) <= max_node_labels:
        return
    else:
    # get the k most frequent node labels from the unique_frequency sorted by frequency
        most_frequent = sorted(unique_frequency, key=unique_frequency.get, reverse=True)[:max_node_labels - 1]
        # add mapping most frequent to 0 to k-1
        mapping = {key: max_node_labels - (value + 2) for key, value in zip(most_frequent, range(max_node_labels))}
        # relabel the node labels
        for g_labels in node_labels:
            for i, l in enumerate(g_labels):
                if l in mapping:
                    g_labels[i] = mapping[l]
                else:
                    g_labels[i] = max_node_labels - 1
    return node_labels




def main():
    data_path = "../../../../GraphData/DS_all/"
    # save_wl_labels(data_path, db_names=['IMDB-BINARY', 'IMDB-MULTI', 'DD', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'])
    #save_wl_labels(data_path, db_names=['MUTAG'])
    #save_circle_labels(data_path, db_names=['SYNTHETICnew'], length_bound=5)
    #save_wl_labels(data_path, db_names=['ZINC'])
    #save_circle_labels(data_path, db_names=['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=8, cycle_type='induced')
    #save_circle_labels(data_path, db_names=['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=10, cycle_type='induced')
    #save_circle_labels(data_path, db_names=['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=6, cycle_type='simple')
    #save_circle_labels(data_path, db_names=['MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=10, cycle_type='simple')
    #save_clique_labels(data_path, db_names=['DHFR', 'MUTAG', 'NCI1', 'NCI109', 'Mutagenicity', 'SYNTHETICnew'], max_clique=6)
    #save_circle_labels(data_path, db_names=['DHFR', 'MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=10, cycle_type='induced')
    #save_circle_labels(data_path, db_names=['DHFR', 'MUTAG', 'NCI1', 'NCI109', 'Mutagenicity'], length_bound=10, cycle_type='simple', max_node_labels=500)
    #save_subgraph_labels(data_path, db_names=['MUTAG'], subgraphs=[nx.cycle_graph(6)])
    #save_circle_labels(data_path, db_names=['DHFR', 'MUTAG'], length_bound=100, cycle_type='induced')
    #save_circle_labels(data_path, db_names=['PROTEINS', 'ENZYMES'], length_bound=6)
    #save_circle_labels(data_path, db_names=['IMDB-MULTI'], length_bound=4, cycle_type='simple')
    #save_circle_labels(data_path, db_names=['ZINC'], length_bound=10, cycle_type='simple')
    save_clique_labels(data_path, db_names=['ZINC'],
                       max_clique=50)




if __name__ == '__main__':
    main()
