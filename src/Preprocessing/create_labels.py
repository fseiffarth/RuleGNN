# generate WL labels for the graph data and save them to a file
import time
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
import torch
from networkx.algorithms.isomorphism import GraphMatcher
from torch_geometric.io import fs

from src.utils import NodeLabeling
from src.utils.GraphData import RuleGNNDataset
from src.utils.GraphLabels import NodeLabels
from src.utils.NodeLabeling import weisfeiler_lehman_node_labeling

def save_labels_to_file(file:Path, dataset_name:str, label_name:str, graph_node_labels:Optional[Union[List[List[int]], torch.Tensor]], max_labels:None):
    """
    Save the node labels to a file
    :param file: Path to the file
    :param dataset_name: Name of the dataset
    :param label_name: Name of the labels
    :param graph_node_labels: List of lists with the node labels for each graph or torch.Tensor with the node labels
    :param max_labels: Maximum number of labels to use
    """
    if isinstance(graph_node_labels, torch.Tensor):
        pass
    elif isinstance(graph_node_labels, list):
        # flatten the node labels
        graph_node_labels = torch.tensor([label for graph_labels in graph_node_labels for label in graph_labels])
    else:
        raise ValueError("graph_node_labels must be either a torch.Tensor or a list of lists")
    # save the node labels to a file as torch tensor with the original labels as first column and the new labels as second column
    fs.torch_save(
        (dataset_name, label_name, relabel_node_labels(graph_node_labels, max_labels)), str(file)
    )

def save_primary_labels(graph_data:RuleGNNDataset, label_path=None, max_labels=None, save_times=None) -> str:
    l = f'primary'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    # save the node labels to a file
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        save_labels_to_file(file,graph_data.name, l, graph_data.node_labels['primary'], max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
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
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    if not file.exists():
        print(f"Saving {l} for {graph_data.name} to {file}")
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        for i,graph in enumerate(graph_data.nx_graphs):
            node_labels.append([0 for _ in range(len(graph.nodes()))])
            for node in graph.nodes():
                node_labels[-1][node] = graph.degree(node)
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels=max_labels)
        #write_node_labels(file, node_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_labeled_degree_labels(graph_data:RuleGNNDataset, label_path=None, max_labels=None, save_times=None)->str:
    # save the node labels to a file
    l = 'wl_labeled_0'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        # iterate over the graphs and get the degree of each node
        node_labels = []
        unique_neighbor_labels = set()
        node_to_hash = dict()
        for graph_id, graph in enumerate(graph_data.nx_graphs):
            for i, node in enumerate(graph.nodes(data=True)):
                neighbors = list(graph.neighbors(node[0]))
                node_identifier = [node[1]['primary_label']]
                node_identifier += [graph.nodes[neighbor]['primary_label'] for neighbor in neighbors]
                # convert to tuple and add to set
                node_identifier = tuple(node_identifier)
                unique_neighbor_labels.add(node_identifier)
                node_to_hash[node[0]] = node_identifier
        # convert the unique neighbor labels to a dict
        unique_neighbor_label_dict = {label: i for i, label in enumerate(unique_neighbor_labels)}
        for graph in graph_data.nx_graphs:
            node_labels.append([unique_neighbor_label_dict[node_to_hash[node]] for node in graph.nodes()])
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_trivial_labels(graph_data:RuleGNNDataset, label_path=None,save_times=None)->str:
    # save the node labels to a file
    l = 'trivial'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        # label 0 for all nodes
        trivial_labels = torch.zeros(len(graph_data.data.x), dtype=torch.long)
        save_labels_to_file(file, graph_data.name, l, trivial_labels, max_labels=None)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def save_index_labels(graph_data:RuleGNNDataset, max_labels=None, label_path=None, save_times=None)->str:
    l = 'index'
    if max_labels is not None:
        l = f'{l}_{max_labels}'
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f"{graph_data.name}_labels_{l}.pt")
    # check whether the file already exists
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        node_labels = []
        start_time = time.time()
        for graph in graph_data.nx_graphs:
            node_labels.append([index for index, node in enumerate(graph.nodes())])
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
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
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        graph_node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=False)
        save_labels_to_file(file, graph_data.name, l, graph_node_labels, max_labels)
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
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
        start_time = time.time()
        node_labels, unique_node_labels, db_unique_node_labels = weisfeiler_lehman_node_labeling(graph_data.nx_graphs, depth=depth, labeled=True)
        save_labels_to_file(file, graph_data.name, l, node_labels, max_labels)
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
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {cycle_type} cycles for {graph_data.name} to {file}")
        start_time = time.time()
        cycle_dict = []
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

        save_labels_to_file(file, graph_data.name, l, labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {cycle_type}_cycles_{length_bound}{l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_in_circle_labels(graph_data:RuleGNNDataset, length_bound=6, max_labels=None, label_path=None, save_times=None)->str:
    l = f'in_cycle_{length_bound}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
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

        save_labels_to_file(file, graph_data.name, l, labels, max_labels=None)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file



def save_subgraph_labels(graph_data:RuleGNNDataset, subgraphs=List[nx.Graph], name='subgraph', subgraph_id=0, max_labels=None, label_path=None, save_times=None)->str:
    l = f'{name}_{subgraph_id}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
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

        save_labels_to_file(file, graph_data.name, l, labels, max_labels=max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file


def save_clique_labels(graph_data:RuleGNNDataset, max_clique=6, max_labels=None, label_path=None, save_times=None)->str:
    l = f'cliques_{max_clique}'
    if max_labels is not None:
        l = f"{l}_{max_labels}"
    if label_path is None:
        raise ValueError("No label path given")
    else:
        file = label_path.joinpath(f'{graph_data.name}_labels_{l}.pt')
    if not file.exists():
        print(f"Saving {l} labels for {graph_data.name} to {file}")
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

        save_labels_to_file(file, graph_data.name, l, labels, max_labels)
        if save_times is not None:
            try:
                with open(save_times, 'a') as f:
                    f.write(f"{graph_data.name}, {l}, {time.time() - start_time}\n")
            except:
                raise ValueError("No save time path given")
    else:
        print(f"File {file} already exists. Skipping.")
    return file

def relabel_node_labels(node_labels: torch.Tensor, max_number_labels:Optional[int]) -> torch.Tensor:
    '''
    Relabel the original labels by mapping them to 0, 1, 2, ... where 0 is the most frequent label of the original labels
    param node_labels: torch.Tensor with the original node labels
    param max_number_labels: Optional[int]
    return: n x 2 torch.Tensor with the original labels as first column and the new labels as second column
    '''
    # get frequency of each value in the new labels
    unique_labels_count = torch.bincount(node_labels)
    # sort the unique labels by the frequency and keep the indices
    sorted_indices = torch.argsort(unique_labels_count, descending=True)
    # if max_number_labels is given, set sorted_indices after max_number_labels to max_number_labels - 1
    # reindex the unique labels: most frequent label is 0, second most frequent is 1, ...
    frequency_sorted_labels = node_labels.new(sorted_indices).argsort()[node_labels]
    if max_number_labels is not None:
        frequency_sorted_labels = torch.where(frequency_sorted_labels >= max_number_labels, max_number_labels - 1, frequency_sorted_labels)
    return torch.stack([node_labels, frequency_sorted_labels], dim=1)