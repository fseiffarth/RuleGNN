import networkx as nx
import torch
from ReadWriteGraphs.GraphDataToGraphList import attributes_to_np_array
import RuleFunctions.Rules as rule


def four_nodes(distance_list, cycle_list):
    graph_list = []
    data = []


    g1 = nx.Graph()
    for i in [1, 2]:
        g1.add_node(i, label=attributes_to_np_array("0"))
    for i in [0, 3]:
        g1.add_node(i, label=attributes_to_np_array("1"))
    g1.add_edges_from([(0, 1), (1, 2), (2, 3)], label=attributes_to_np_array("0"))
    g2 = nx.Graph()
    for i in [0, 2, 3]:
        g2.add_node(i, label=attributes_to_np_array("0"))
    g2.add_node(1, label=attributes_to_np_array("2"))
    g2.add_edges_from([(0, 1), (1, 2), (1, 3)], label=attributes_to_np_array("0"))
    g3 = nx.Graph()
    for i in [0, 1, 2, 3]:
        g3.add_node(i, label=attributes_to_np_array("1"))
    g3.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)], label=attributes_to_np_array("0"))
    g4 = nx.Graph()
    for i in [0, 1]:
        g4.add_node(i, label=attributes_to_np_array("1"))
    g4.add_node(2, label=attributes_to_np_array("2"))
    g4.add_node(3, label=attributes_to_np_array("0"))
    g4.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 0)], label=attributes_to_np_array("0"))

    graph_list = [g1, g2, g3, g4]
    labels = [1, 1, -1, -1]
    for x in range(10):
        graph_list += [g1, g2, g3, g4]
        labels += [1, 1, -1, -1]
    for graph in graph_list:
        data.append(torch.ones((graph.number_of_nodes(), 1), dtype=torch.double))
        distance_list.append(dict(nx.all_pairs_shortest_path_length(graph)))
        cycle_list.append(rule.generate_cycle_list(graph))
    """
    g1 = nx.Graph()
    for i in [0,1,2]:
        g1.add_node(i, label=attributes_to_np_array("0"))
    g1.add_edges_from([(0, 1), (1, 2)], label=attributes_to_np_array("0"))
    g2 = nx.Graph()
    for i in [0,1,2]:
        g2.add_node(i, label=attributes_to_np_array("0"))
    g2.add_edges_from([(0, 1), (1, 2), (2,0)], label=attributes_to_np_array("0"))
    graph_list = [g1, g2]
    labels = [1, -1]
    for x in range(9):
        graph_list += [g1, g2]
        labels += [1, -1]
    for graph in graph_list:
        data.append(torch.ones((graph.number_of_nodes(), 1), dtype=torch.double))
        distance_list.append(dict(nx.all_pairs_shortest_path_length(graph)))
        cycle_list.append(rule.generate_cycle_list(graph))
    """
    return data, labels, (graph_list, labels, [])
