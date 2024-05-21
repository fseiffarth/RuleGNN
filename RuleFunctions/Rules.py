'''
Created on 14.03.2019

@author:
'''

import ReadWriteGraphs.GraphDataToGraphList as rg
import networkx as nx
import numpy as np



def matrix_to_list_index(pos_x, pos_y, n, m):
    return pos_x * m + pos_y


def list_to_matrix_index(pos, n, m):
    pos_x = pos // m
    pos_y = pos % m
    return (pos_x, pos_y)


def rule_fc(row, column, num_rows, num_cols):
    return row * num_cols + column


def matrix_pos_to_node_feature(pos_x, pos_y, in_features, node_num):
    x_node = (pos_x // in_features) % node_num
    y_node = pos_y // in_features
    x_feature = pos_x % in_features
    y_feature = pos_y % in_features
    kernel_pos = pos_x // (in_features * node_num)
    return (x_node, y_node, x_feature, y_feature, kernel_pos)


def bias_pos_to_node_feature(pos_x, in_features, node_num):
    x_node = (pos_x // in_features) % node_num
    x_feature = pos_x % in_features
    kernel_pos = pos_x // (in_features * node_num)
    return (x_node, x_feature, kernel_pos)


def weight_rule_graphs(row, column, in_features, n_node_labels, n_edge_labels, n_kernels, graph):
    (x_node, y_node, x_feature, y_feature, kernel_pos) = matrix_pos_to_node_feature(row, column, in_features,
                                                                                    graph.number_of_nodes())
    if x_feature == y_feature:
        if (graph.has_edge(x_node, y_node) or x_node == y_node) and max(int(rg.node_label_vector(graph, x_node)), int(
                rg.node_label_vector(graph, y_node))) < n_node_labels:
            x_label = rg.node_label_vector(graph, x_node)
            y_label = rg.node_label_vector(graph, y_node)
            e_label = rg.edge_label_vector(graph, x_node, y_node)
            if not e_label or n_edge_labels == 1:
                e_label = 0
            pos = matrix_to_list_index(int(x_label) + int(x_feature) * n_node_labels + int(
                e_label) * n_node_labels * in_features + kernel_pos * n_edge_labels * in_features * n_node_labels,
                                       int(y_label), n_node_labels * in_features * n_edge_labels * n_kernels,
                                       n_node_labels)
            return pos
        else:
            return -1
    else:
        return -1


# looks only at real edges
def weight_rule_graphs_edge(row, column, in_features, n_node_labels, n_edge_labels, n_kernels, graph):
    (x_node, y_node, x_feature, y_feature, kernel_pos) = matrix_pos_to_node_feature(row, column, in_features,
                                                                                    graph.number_of_nodes())
    if x_feature == y_feature:
        if x_node == y_node and max(int(rg.node_label_vector(graph, x_node)),
                                    int(rg.node_label_vector(graph, y_node))) < n_node_labels:
            y_label = rg.node_label_vector(graph, y_node)
            pos = matrix_to_list_index(int(x_feature), int(y_label),
                                       n_node_labels * in_features * n_edge_labels * n_kernels, n_node_labels)
            return pos
        elif graph.has_edge(x_node, y_node) and max(int(rg.node_label_vector(graph, x_node)),
                                                    int(rg.node_label_vector(graph, y_node))) < n_node_labels:
            x_label = rg.node_label_vector(graph, x_node)
            y_label = rg.node_label_vector(graph, y_node)
            e_label = rg.edge_label_vector(graph, x_node, y_node)
            if not e_label or n_edge_labels == 1:
                e_label = 0
            pos = matrix_to_list_index(int(x_label) + int(x_feature) * n_node_labels + int(
                e_label) * n_node_labels * in_features + kernel_pos * n_edge_labels * in_features * n_node_labels + in_features,
                                       int(y_label), n_node_labels * in_features * n_edge_labels * n_kernels,
                                       n_node_labels)
            return pos
        else:
            return -1
    else:
        return -1


# weight rule by node and edge labels
def weight_rule_wf(n1, n2, graph_data, graph_id: int, *args, **kwargs):
    x_label = -1
    y_label = -1
    e_label = -1

    self_loop = False
    if 'self_loop' in kwargs and kwargs['self_loop'] == True:
        self_loop = kwargs['self_loop']

    if n1 == n2:
        if self_loop:
            x_label = graph_data.secondary_node_labels.node_labels[graph_id][n1]
            y_label = graph_data.secondary_node_labels.node_labels[graph_id][n2]
            e_label = 0
        return x_label, y_label, e_label, 0
    elif graph_data.graphs[graph_id].has_edge(n1, n2):
        x_label = graph_data.secondary_node_labels.node_labels[graph_id][n1]
        y_label = graph_data.secondary_node_labels.node_labels[graph_id][n2]
        e_label = rg.edge_label_vector(graph_data.graphs[graph_id], n1, n2)
    return x_label, y_label, e_label, 0


# weight rule by node and edge labels and node degree
def weight_rule_wf_deg(n1, n2, graph, *args):
    x_label = -1
    y_label = -1
    e_label = -1
    deg = 0
    if n1 == n2:
        x_label = rg.node_label_vector(graph, n1)
        y_label = rg.node_label_vector(graph, n2)
        e_label = 0
        deg = 0
    elif graph.has_edge(n1, n2):
        x_label = rg.node_label_vector(graph, n1)
        y_label = rg.node_label_vector(graph, n2)
        e_label = rg.edge_label_vector(graph, n1, n2)
        deg = graph.degree(n1) + graph.degree(n2)
    return x_label, y_label, e_label, deg


# weight rule by node labels and node distance
def weight_rule_wf_dist(n1, n2, graph_data, node_labels, graph_id):
    if n1 == n2:
        dist = 0
    else:
        if n2 in graph_data.distance_list[graph_id][n1]:
            dist = graph_data.distance_list[graph_id][n1][n2]
        else:
            dist = -1
    x_label = node_labels.node_labels[graph_id][n1]
    y_label = node_labels.node_labels[graph_id][n2]
    e_label = 0
    return x_label, y_label, e_label, dist


# weight rule by node label and common cycles
def weight_rule_wf_cycle(n1, n2, graph, cycle_list, counter):
    x_label = -1
    y_label = -1
    e_label = -1
    cycle_length = 0
    if n1 == n2:
        x_label = rg.node_label_vector(graph, n1)
        y_label = rg.node_label_vector(graph, n2)
        e_label = 0
        cycle_length = 0
    else:
        cycle_length = cycle_list[counter][n1][n2]
        if cycle_length != 0:
            x_label = rg.node_label_vector(graph, n1)
            y_label = rg.node_label_vector(graph, n2)
            e_label = 0
    return x_label, y_label, e_label, cycle_length


# weight rule by node label
def node_label_rule(n1, labels, graph_id):
    return labels.node_labels[graph_id][n1]


def weight_rule_distances(row, column, in_features, n_node_labels, max_distance, n_kernels, graph, distance_list):
    (x_node, y_node, x_feature, y_feature, kernel_pos) = matrix_pos_to_node_feature(row, column, in_features,
                                                                                    graph.number_of_nodes())
    if x_feature == y_feature:
        if max(int(rg.node_label_vector(graph, x_node)), int(rg.node_label_vector(graph, y_node))) < n_node_labels:
            x_label = rg.node_label_vector(graph, x_node)
            y_label = rg.node_label_vector(graph, y_node)
            distance = distance_list[x_node][y_node]
            if distance < max_distance:
                pos = matrix_to_list_index(int(x_label) + int(
                    x_feature) * n_node_labels + distance * n_node_labels * in_features + kernel_pos * max_distance * in_features * n_node_labels,
                                           int(y_label), n_node_labels * in_features * max_distance * n_kernels,
                                           n_node_labels)
            else:
                pos = -1
            return pos
        else:
            return -1
    else:
        return -1


def weight_rule_cycles(row, column, in_features, n_node_labels, max_cycle_lenth, n_kernels, graph, cycle_dat):
    (x_node, y_node, x_feature, y_feature, kernel_pos) = matrix_pos_to_node_feature(row, column, in_features,
                                                                                    graph.number_of_nodes())
    if x_feature == y_feature:
        if max(int(rg.node_label_vector(graph, x_node)), int(rg.node_label_vector(graph, y_node))) < n_node_labels:
            x_label = rg.node_label_vector(graph, x_node)
            y_label = rg.node_label_vector(graph, y_node)
            cycle_length = cycle_dat[x_node][y_node]
            if cycle_length < max_cycle_lenth and cycle_length != 0:
                pos = matrix_to_list_index(int(x_label) + int(
                    x_feature) * n_node_labels + cycle_length * n_node_labels * in_features + kernel_pos * cycle_length * in_features * n_node_labels,
                                           int(y_label), n_node_labels * in_features * max_cycle_lenth * n_kernels,
                                           n_node_labels)
            else:
                pos = -1
            return pos
        else:
            return -1
    else:
        return -1


def generate_cycle_list(G):
    cycle_list = nx.minimum_cycle_basis(G)
    W = np.zeros((G.number_of_nodes(), G.number_of_nodes()), dtype=np.int16)

    for i in range(0, W.size):
        for j in range(0, W[0].size):
            for t in cycle_list:
                if i in t and j in t:
                    W[i][j] = len(t)
    return W


def bias_rule_graphs(row, in_features, n_node_labels, n_kernels, graph):
    (x_node, x_feature, kernel_pos) = bias_pos_to_node_feature(row, in_features, graph.number_of_nodes())
    x_label = rg.node_label_vector(graph, x_node)
    pos = matrix_to_list_index(int(x_feature) + kernel_pos * in_features, int(x_label), in_features * n_kernels,
                               n_node_labels)
    return pos


def w_resize_distribution_rule(row, column, out_features, col_number, n_node_labels, graph):
    k_pos = row
    node = column // (col_number // graph.number_of_nodes())
    feature = column % (col_number // graph.number_of_nodes())
    node_label = rg.node_label_vector(graph, node)
    pos = matrix_to_list_index(feature + k_pos * (col_number // graph.number_of_nodes()), int(node_label), out_features,
                               n_node_labels)
    return pos


def graph_to_weight_pos(graph, rule):
    result = np.zeros()


def graph_list_to_weight_pos(graph, rule):
    t = np.zeros()
