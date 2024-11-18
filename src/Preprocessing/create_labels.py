# generate WL labels for the graph data and save them to a file
import time
from pathlib import Path
from typing import List, Optional

import networkx as nx
import torch
from networkx.algorithms.isomorphism import GraphMatcher

from src.utils import NodeLabeling
from src.utils.GraphData import RuleGNNDataset
from src.utils.GraphLabels import NodeLabels
from src.utils.NodeLabeling import weisfeiler_lehman_node_labeling

def save_labels_to_file(file:Path, graph_node_labels:List[List[int]], max_labels:None):
    # flatten the node labels
    node_labels = [label for graph_labels in graph_node_labels for label in graph_labels]
    node_labels, relabeled_node_labels = relabel_node_labels(node_labels, max_labels)
    # save the node labels to a file as torch tensor with the original labels as first column and the new labels as second column
    label_tensor = torch.stack([torch.tensor(node_labels), torch.tensor(relabeled_node_labels)], dim=1)
    torch.save(label_tensor, file)

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

def save_primary_labels(graph_data:RuleGNNDataset, label_path=None, max_labels=None, save_times=None) -> str:
    l = f'primary'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    # save the node labels to a file
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_{l}_labels.pt")
    if not file.exists():
        start_time = time.time()
        print(f"Saving primary labels for {graph_data.name} to {file}")
        node_labels = [x.item() for x in graph_data.primary_node_labels]
        node_labels, relabeled_node_labels = relabel_node_labels(node_labels, max_labels)
        # save the node labels to a file as torch tensor with the original labels as first column and the new labels as second column
        label_tensor = torch.stack([torch.tensor(node_labels), torch.tensor(relabeled_node_labels)], dim=1)
        torch.save(label_tensor, file)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, primary, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

    node_labels = graph_data.node_labels['primary'].node_labels
    node_labels = relabel_node_labels(node_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_primary_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving primary labels for {graph_data.name} to {file}")
        start_time = time.time()
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, primary, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_degree_labels(graph_data:RuleGNNDataset, label_path=None, max_labels=None, save_times=None)->str:
    #save the node labels to a file
    l = 'wl_0'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_{l}_labels.pt")
    if not file.exists():
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        for i,graph in enumerate(graph_data.nx_graphs):
            node_labels.append([0 for _ in range(len(graph.nodes()))])
            for node in graph.nodes():
                node_labels[-1][node] = graph.degree(node)
        print(f"Saving wl_0 labels for {graph_data.name} to {file}")
        save_labels_to_file(file, node_labels, max_labels=max_labels)
        #write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, wl_0, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_labeled_degree_labels(graph_data:RuleGNNDataset, label_path=None, save_times=None)->str:
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_wl_labeled_0_labels.pt")
    # check whether the file already exists
    if not file.exists():
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        unique_neighbor_labels = set()
        node_to_hash = dict()
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            for i, node in enumerate(graph.nodes()):
                neighbors = list(graph.neighbors(node))
                node_identifier = [graph_data.node_labels['primary'].node_labels[graph_id][i]]
                node_identifier += [graph_data.node_labels['primary'].node_labels[graph_id][neighbor] for neighbor in neighbors]
                node_neighbor_labels = [graph_data.node_labels['primary'].node_labels[i]] + [graph_data.node_labels['primary'].node_labels[neighbor] for neighbor in graph.neighbors(node)]
                # convert to tuple and add to set
                node_identifier = tuple(node_identifier)
                unique_neighbor_labels.add(node_identifier)
                node_to_hash[node] = node_identifier
        # convert the unique neighbor labels to a dict
        unique_neighbor_label_dict = {label: i for i, label in enumerate(unique_neighbor_labels)}

        for graph in graph_data.nx_graphs:
            node_labels += [unique_neighbor_label_dict[node_to_hash[node]] for node in graph.nodes()]
            #node_labels.append([unique_neighbor_label_dict[node_to_hash[node]] for node in graph.nodes()])
        node_labels = relabel_node_labels(node_labels)

        print(f"Saving wl_labeled_0 labels for {graph_data.name} to {file}")
        torch.save(node_labels, file)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, wl_labeled_0, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_trivial_labels(graph_data:RuleGNNDataset, label_path=None,save_times=None)->str:
    # save the node labels to a file
    # save node_labels as numpy array
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_trivial_labels.txt')
    if not file.exists():
        print(f"Saving trivial labels for {graph_data.name} to {file}")
        start_time = time.time()
        node_labels = graph_data.node_labels['primary'].node_labels
        # label 0 for all nodes
        node_labels = [[0 for _ in range(len(g_labels))] for g_labels in node_labels]

        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, trivial, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_node_labels(graph_data: RuleGNNDataset, labels, label_path:Path, label_string, max_labels=None, save_times=None):
    start_time = time.time()
    if max_labels  and max_labels < len(labels):
        labels = relabel_most_frequent_node_labels(labels, max_labels)
    # save the node labels to a file
    # save node_labels as numpy array
    file = label_path.joinpath(f"{graph_data.name}_{label_string}_labels.txt")
    write_node_labels(file, labels)
    if save_times is not None:
        try:
            with open(save_times, 'a') as f:
                f.write(f"{graph_data.name}, primary, {time.time() - start_time}\n")
        except:
            raise ValueError("No save time path given")

def save_index_labels(graph_data:RuleGNNDataset, max_labels=None, label_path=None, save_times=None)->str:
    node_labels = []
    start_time = time.time()
    for graph in graph_data.nx_graphs:
        node_labels.append([index for index, node in enumerate(graph.nodes())])
    if max_labels is not None:
        node_labels = relabel_most_frequent_node_labels(node_labels, max_labels)
    # save the node labels to a file
    if label_path is None:
        raise ValueError("No label path given")
    else:
        if max_labels is not None:
            file = label_path.joinpath(f"{graph_data.name}_index_{max_labels}_labels.txt")
        else:
            file = label_path.joinpath(f"{graph_data.name}_index_labels.txt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving primary labels for {graph_data.name} to {file}")
        write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, index, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_wl_labels(graph_data:RuleGNNDataset, depth, max_labels=None, label_path=None, save_times=None)->str:
    # save the node labels to a file
    l = f'wl_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_{l}_labels.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        graph_node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=False)
        save_labels_to_file(file, graph_node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_wl_labeled_labels(graph_data:RuleGNNDataset, depth, max_labels=None, label_path=None, save_times=None)->str:
    # save the node labels to a file
    l = f'wl_labeled_{depth}'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_{l}_labels.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=True)
        save_labels_to_file(file, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}_{max_labels}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_cycle_labels(graph_data:RuleGNNDataset, length_bound=6, max_labels=None, cycle_type='simple', label_path=None, save_times=None)->str:
    if cycle_type not in ['simple', 'induced']:
        raise ValueError("Cycle type must be either 'simple' or 'induced'")
    l = 'simple_cycles'
    if cycle_type == 'induced':
        l = 'induced_cycles'
    l = f'{l}_{length_bound}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_{l}_labels.pt')
    if not file.exists():
        start_time = time.time()
        cycle_dict = []
        print(f"Saving {cycle_type} cycles for {graph_data.name} to {file}")
        for graph in graph_data.nx_graphs:
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
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            labels.append([])
            for node in graph.nodes():
                if node in cycle_dict[graph_id]:
                    cycle_d = str(cycle_dict[graph_id][node])
                    labels[-1].append(label_dict[cycle_d])
                else:
                    labels[-1].append(len(label_dict))

        save_labels_to_file(file, labels, max_labels)
        #write_node_labels(file, labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {cycle_type}_cycles_{length_bound}{l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_in_circle_labels(graph_data:RuleGNNDataset, length_bound=6, label_path=None, save_times=None)->str:
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_cycles_{length_bound}_labels.txt')
    if not file.exists():
        print(f"Saving in circle labels for {graph_data.name} to {file}")
        start_time = time.time()
        node_in_cycle = []
        for graph in graph_data.nx_graphs:
            node_in_cycle.append({})
            cycles = nx.chordless_cycles(graph, length_bound)
            for cycle in cycles:
                for node in cycle:
                    node_in_cycle[-1][node] = 1


        # set the node labels, if node is in a cycle label 1, else 0
        labels = []
        for graph_id, graph in enumerate(graph_data.nx_graphs):
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
                    f.write(f"{graph_data.name}, cycles_{length_bound}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_subgraph_labels(graph_data:RuleGNNDataset, subgraphs=List[nx.Graph], name='subgraph', id=0, label_path=None, save_times=None)->str:
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_{name}_{id}_labels.txt')
    if not file.exists():
        start_time = time.time()
        subgraph_dict = []
        for i, graph in enumerate(graph_data.nx_graphs):
            # print the progress
            print(f"Graph {i + 1}/{len(graph_data.nx_graphs)}")
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
        for graph_id, graph in enumerate(graph_data.nx_graphs):
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
                    f.write(f"{graph_data.name}, {name}_{id}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_clique_labels(graph_data:RuleGNNDataset, max_clique=6, max_labels=None, label_path=None, save_times=None)->str:
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_cliques_{max_clique}_labels.txt')
    if not file.exists():
        start_time = time.time()
        clique_dict = []
        for graph in graph_data.nx_graphs:
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
        for graph_id, graph in enumerate(graph_data.nx_graphs):
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
                    f.write(f"{graph_data.name}, cliques_{max_clique}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def relabel_node_labels(node_labels: List[int], max_number_labels:Optional[int]) -> (List[int], List[int]):
    '''
    Relabel the original labels by mapping them to 0, 1, 2, ... where 0 is the most frequent label of the original labels
    param node_labels: List[int]
    param max_number_labels: Optional[int]
    return: List[int], List[int] pair of the original labels and the new labels
    '''
    new_labels = []
    # get the unique labels
    unique_labels: dict[int, int] = {}
    for label in node_labels:
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
        # take into account the max number of labels
        if max_number_labels is not None:
            if i >= max_number_labels:
                new_label_mapping[label] = max_number_labels - 1
    for label in node_labels:
        new_labels.append(new_label_mapping[label])
    return node_labels, new_labels