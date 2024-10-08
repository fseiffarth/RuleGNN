# generate WL labels for the graph data and save them to a file
import time
from pathlib import Path
from typing import List

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from src.utils import GraphData, NodeLabeling
from src.utils.GraphData import relabel_most_frequent
from src.utils.GraphLabels import NodeLabels
from src.utils.NodeLabeling import weisfeiler_lehman_node_labeling


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

def save_primary_labels(graph_data:GraphData, label_path=None, save_times=None):
    node_labels = graph_data.node_labels['primary'].node_labels
    node_labels = relabel_node_labels(node_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.graph_db_name}_primary_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving primary labels for {graph_data.graph_db_name} to {file}")
        start_time = time.time()
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, primary, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")



def save_degree_labels(graph_data:GraphData, label_path=None, save_times=None):
    start_time = time.time()
    # iterate over the graphs and get the degree of each node
    node_labels = []
    for i,graph in enumerate(graph_data.graphs):
        node_labels.append([0 for _ in range(len(graph.nodes()))])
        for node in graph.nodes():
            node_labels[-1][node] = graph.degree(node)
    node_labels = relabel_node_labels(node_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.graph_db_name}_wl_0_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving wl_0 labels for {graph_data.graph_db_name} to {file}")
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, wl_0, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")

def save_labeled_degree_labels(graph_data:GraphData, label_path=None, save_times=None):
    start_time = time.time()
    # iterate over the graphs and get the degree of each node
    node_labels = []
    unique_neighbor_labels = set()
    node_to_hash = dict[int, tuple]
    for graph in graph_data.graphs:
        for i, node in enumerate(graph.nodes()):
            node_neighbor_labels = [graph_data.node_labels['primary'].node_labels[i]] + [graph_data.node_labels['primary'].node_labels[neighbor] for neighbor in graph.neighbors(node)]
            # convert to tuple and add to set
            node_neighbor_labels = tuple(node_neighbor_labels)
            unique_neighbor_labels.add(node_neighbor_labels)
            node_to_hash[node] = node_neighbor_labels
    # convert the unique neighbor labels to a dict
    unique_neighbor_label_dict = {label: i for i, label in enumerate(unique_neighbor_labels)}

    for graph in graph_data.graphs:
        node_labels.append([unique_neighbor_label_dict[node_to_hash[node]] for node in graph.nodes()])

    node_labels = relabel_node_labels(node_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.graph_db_name}_wl_0_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving wl_0 labels for {graph_data.graph_db_name} to {file}")
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, wl_0, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_trivial_labels(graph_data:GraphData, label_path=None,save_times=None):
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_trivial_labels.txt')
    if not file.exists():
        print(f"Saving trivial labels for {graph_data.graph_db_name} to {file}")
        start_time = time.time()
        node_labels = graph_data.node_labels['primary'].node_labels
        # label 0 for all nodes
        node_labels = [[0 for _ in range(len(g_labels))] for g_labels in node_labels]

        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, trivial, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_node_labels(graph_data: GraphData, labels, label_path:Path, label_string, max_labels=None, save_times=None):
    start_time = time.time()
    if max_labels  and max_labels < len(labels):
        labels = relabel_most_frequent_node_labels(labels, max_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    file = label_path.joinpath(f"{graph_data.graph_db_name}_{label_string}_labels.txt")
    write_node_labels(file, labels)
    if save_times is not None:
        try:
            with open(save_times, 'a') as f:
                f.write(f"{graph_data.graph_db_name}, primary, {time.time() - start_time}\n")
        except:
            raise ValueError("No save time path given")

def save_index_labels(graph_data:GraphData, max_labels=None, label_path=None, save_times=None):
    node_labels = []
    start_time = time.time()
    for graph in graph_data.graphs:
        node_labels.append([index for index, node in enumerate(graph.nodes())])
    if max_labels is not None:
        node_labels = relabel_most_frequent_node_labels(node_labels, max_labels)
    # save the node labels to a file
    if label_path is None:
        raise ValueError("No label path given")
    else:
        if max_labels is not None:
            file = label_path.joinpath(f"{graph_data.graph_db_name}_index_{max_labels}_labels.txt")
        else:
            file = label_path.joinpath(f"{graph_data.graph_db_name}_index_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving primary labels for {graph_data.graph_db_name} to {file}")
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, index, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")

def save_wl_labeled_labels(graph_data:GraphData, depth = 3, max_labels=None, label_path=None, save_times=None):
    # save the node labels to a file
    l = f'wl_labeled_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_{l}_labels.txt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.graph_db_name} to {file}")
        start_time = time.time()
        node_labeling = NodeLabels()
        node_labeling.node_labels, node_labeling.unique_node_labels, node_labeling.db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.graphs, depth, labeled=True)
        node_labeling.num_unique_node_labels = max(1, len(node_labeling.db_unique_node_labels))
        if max_labels is not None and max_labels > 0:
            l = f'{l}_{max_labels}'

        relabel_most_frequent(node_labeling, max_labels)
        write_node_labels(file, node_labeling.node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_wl_labels(graph_data:GraphData, depth, max_labels=None, label_path=None, save_times=None):
    # save the node labels to a file
    l = f'wl_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_{l}_labels.txt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.graph_db_name} to {file}")
        start_time = time.time()
        graph_data.add_node_labels(node_labeling_name=l, max_labels=max_labels,
                                   node_labeling_method=NodeLabeling.weisfeiler_lehman_node_labeling,
                                   depth=depth)
        node_labels = graph_data.node_labels[f'{l}_{max_labels}'].node_labels

        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_cycle_labels(graph_data:GraphData, length_bound=6, max_labels=None, cycle_type='simple', label_path=None, save_times=None):
    start_time = time.time()
    cycle_dict = []
    max_labels_str = ''
    if max_labels is not None:
        max_labels_str = f"_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        if cycle_type == 'simple':
            file = label_path.joinpath(f'{graph_data.graph_db_name}_simple_cycles_{length_bound}{max_labels_str}_labels.txt')
        elif cycle_type == 'induced':
            file = label_path.joinpath(f'{graph_data.graph_db_name}_induced_cycles_{length_bound}{max_labels_str}_labels.txt')
    if not file.exists():
        print(f"Saving {cycle_type} cycles for {graph_data.graph_db_name} to {file}")
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

        if max_labels is not None:
            relabel_most_frequent_node_labels(labels, max_labels)

        write_node_labels(file, labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, {cycle_type}_cycles_{length_bound}{max_labels_str}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_in_circle_labels(graph_data:GraphData, length_bound=6, label_path=None, save_times=None):

    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_cycles_{length_bound}_labels.txt')
    if not file.exists():
        print(f"Saving in circle labels for {graph_data.graph_db_name} to {file}")
        start_time = time.time()
        node_in_cycle = []
        for graph in graph_data.graphs:
            node_in_cycle.append({})
            cycles = nx.chordless_cycles(graph, length_bound)
            for cycle in cycles:
                for node in cycle:
                    node_in_cycle[-1][node] = 1


        # set the node labels, if node is in a cycle label 1, else 0
        labels = []
        for graph_id, graph in enumerate(graph_data.graphs):
            labels.append([])
            for node in graph.nodes():
                if node in node_in_cycle[graph_id]:
                    labels[-1].append(1)
                else:
                    labels[-1].append(0)


        write_node_labels(file, labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, cycles_{length_bound}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")



def save_subgraph_labels(graph_data:GraphData, subgraphs=List[nx.Graph], name='subgraph', id=0, label_path=None, save_times=None):
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_{name}_{id}_labels.txt')
    if not file.exists():
        start_time = time.time()
        subgraph_dict = []
        for i, graph in enumerate(graph_data.graphs):
            # print the progress
            print(f"Graph {i + 1}/{len(graph_data.graphs)}")
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

        labels = relabel_node_labels(labels)
        write_node_labels(file, labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, {name}_{id}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def save_clique_labels(graph_data:GraphData, max_clique=6, max_labels=None, label_path=None, save_times=None):
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.graph_db_name}_cliques_{max_clique}_labels.txt')
    if not file.exists():
        start_time = time.time()
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

        labels = relabel_node_labels(labels)
        write_node_labels(file, labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.graph_db_name}, cliques_{max_clique}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")


def relabel_most_frequent_node_labels(node_labels, max_labels):
    """
    Relabel the node labels with the most frequent labels.
    :param node_labels: list of lists
    :param max_labels: int
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
    if len(unique_frequency) <= max_labels:
        return
    else:
        # get the k most frequent node labels from the unique_frequency sorted by frequency
        most_frequent = sorted(unique_frequency, key=unique_frequency.get, reverse=True)[:max_labels - 1]
        # add mapping most frequent to 0 to k-1
        mapping = {key: max_labels - (value + 2) for key, value in zip(most_frequent, range(max_labels))}
        # relabel the node labels
        for g_labels in node_labels:
            for i, l in enumerate(g_labels):
                if l in mapping:
                    g_labels[i] = mapping[l]
                else:
                    g_labels[i] = max_labels - 1
    return node_labels

def relabel_node_labels(node_labels: List[List[int]]) -> List[List[int]]:
    '''
    Relabel the original labels by mapping them to 0, 1, 2, ... where 0 is the most frequent label of the original labels
    '''
    new_labels = []
    # get the unique labels
    unique_labels: dict[int, int] = {}
    for node_label in node_labels:
        for label in node_label:
            if label not in unique_labels:
                unique_labels[label] = 1
            else:
                unique_labels[label] += 1
    # sort the unique labels by the value
    unique_labels = dict(sorted(unique_labels.items(), key=lambda item: item[1], reverse=True))
    # new label mapping
    new_label_mapping: dict[int, int] = {}
    for i, label in enumerate(unique_labels):
        new_label_mapping[label] = i
    for graph_labels in node_labels:
        new_labels.append([new_label_mapping[label] for label in graph_labels])
    return new_labels