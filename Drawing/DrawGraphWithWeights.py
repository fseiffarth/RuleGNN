import click
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from GraphData import NodeLabeling
from GraphData.GraphData import GraphData
from NeuralNetArchitectures import GraphNN
from TrainTestData import TrainTestData as ttd

def draw_graph(graph_data:GraphData, graph_id, ax, node_size=50, node_color='blue', edge_color='black', edge_width=0.5,):
    graph = graph_data.graphs[graph_id]
    node_labels = {}
    for node in graph.nodes():
        key = int(node)
        value = str(graph_data.secondary_node_labels.node_labels[graph_id][node])
        node_labels[key] = f'{value}'
    edge_labels = {}
    for (key1, key2, value) in graph.edges(data=True):
        if "label" in value:
            edge_labels[(key1, key2)] = int(value["label"])
        else:
            edge_labels[(key1, key2)] = ""

    # draw the graph
    pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_color, node_size=node_size)
    nx.draw_networkx_labels(graph, pos=pos, labels=node_labels, ax=ax, font_size=8, font_color='white')
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='black')


def draw_graph_layer(graph_data: GraphData, graph_id, layer, ax, cmap='seismic', node_size=50, edge_width=5):
    weight = layer.get_weights()
    bias = layer.get_bias()
    layer_id = layer.layer_id
    # get the adjacency matrices + the bias vector for the first graph
    # load from txt file
    path = f'../Results/{graph_data.graph_db_name}/Weights/graph_{graph_id}_layer_{layer_id}_parameterWeightMatrix.txt'
    # read as csv with pandas
    df = pd.read_csv(path, sep=';', header=None)

    graph = graph_data.graphs[graph_id]
    weight_matrix = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    # iterate over the rows and columns of the weight matrix
    for i in range(0, graph.number_of_nodes()):
        for j in range(0, graph.number_of_nodes()):
            # check if the edge i,j exists
            if graph.has_edge(i, j):
                weight_matrix[i][j] = weight[df.iloc[i, j]]
    bias_vector = np.asarray(bias)

    weight_min = np.min(weight_matrix)
    weight_max = np.max(weight_matrix)
    bias_min = np.min(bias_vector)
    bias_max = np.max(bias_vector)

    weight_max_abs = max(abs(weight_min), abs(weight_max))
    bias_max_abs = max(abs(bias_min), abs(bias_max))

    # use seismic colormap with maximum and minimum values from the weight matrix
    cmap = plt.get_cmap(cmap)
    # normalize item number values to colormap
    norm_weight = matplotlib.colors.Normalize(vmin=-weight_max_abs, vmax=weight_max_abs)
    norm_bias = matplotlib.colors.Normalize(vmin=-bias_max_abs, vmax=bias_max_abs)
    weight_colors = cmap(norm_weight(weight_matrix))
    bias_colors = cmap(norm_bias(bias_vector))

    # draw the graph
    pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    # graph to digraph
    graph = nx.DiGraph(graph)
    curved_edges = [edge for edge in graph.edges() if reversed(edge) in graph.edges()]
    curved_edges_colors = []
    edge_widths = []
    for edge in curved_edges:
        curved_edges_colors.append(weight_colors[edge[0]][edge[1]])
        edge_widths.append(edge_width * abs(weight_matrix[edge[0]][edge[1]]) / weight_max_abs)
    arc_rad = 0.25
    nx.draw_networkx_edges(graph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors, width=edge_widths,
                           connectionstyle=f'arc3, rad = {arc_rad}', arrowsize=5)

    node_colors = []
    node_sizes = []
    for node in graph.nodes():
        node_label = graph_data.secondary_node_labels.node_labels[graph_id][node]
        node_colors.append(bias_colors[node_label])
        node_sizes.append(node_size * abs(bias_vector[node_label]) / bias_max_abs)

    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)


@click.command()
@click.option('--data_path', default="../../GraphData/DS_all/", help='Path to the graph data')
@click.option('--db', default="MUTAG", help='Database to use')
def main(data_path, db):
    run = 1
    k_val = 0
    kFold = 10
    # load a model from Mutag
    graph_data = GraphData()
    graph_data.init_from_graph_db(data_path, db, with_distances=False, with_cycles=False)
    graph_data.secondary_node_labels, graph_data.secondary_edge_labels = graph_data.add_node_labels(
        NodeLabeling.weisfeiler_lehman_node_labeling)

    model_path = f'../Results/{db}/Models/model_run_{run}_val_step_{k_val}.pt'
    seed = k_val + kFold * run
    run_test_indices = ttd.get_data_indices(graph_data.num_graphs, seed=run, kFold=kFold)
    # check if the model exists
    try:
        with open(model_path, 'r'):
            training_data, validate_data, test_data = ttd.get_train_validation_test_list(test_indices=run_test_indices,
                                                                                         validation_step=k_val,
                                                                                         seed=seed,
                                                                                         balanced=True,
                                                                                         graph_labels=graph_data.graph_labels,
                                                                                         val_size=0)
            # load the model and evaluate the performance on the test data
            net = GraphNN.GraphNetOriginal(graph_data=graph_data, n_node_features=1,
                                           n_node_labels=graph_data.secondary_node_labels.num_unique_node_labels,
                                           n_edge_labels=graph_data.primary_edge_labels.num_unique_edge_labels,
                                           seed=seed,
                                           dropout=0,
                                           out_classes=graph_data.num_classes,
                                           print_weights=False)
            net.load_state_dict(torch.load(model_path))
            # evaluate the performance of the model on the test data
            outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
            with torch.no_grad():
                for j, data_pos in enumerate(test_data, 0):
                    inputs = torch.DoubleTensor(graph_data.inputs[data_pos])
                    outputs[j] = net(inputs, data_pos)
                labels = graph_data.one_hot_labels[test_data]
                # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                counter = 0
                correct = 0
                for i, x in enumerate(outputs, 0):
                    if torch.argmax(x) == torch.argmax(labels[i]):
                        correct += 1
                    counter += 1
                accuracy = correct / counter
                print(f"Accuracy for model {model_path} is {accuracy}")


            graph_ids = [0, 10, 100]
            rows = len(graph_ids)
            cols = 4
            fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3*cols, 3*rows))
            plt.subplots_adjust(wspace=0, hspace=0)
            layers = [net.l1, net.l2, net.l3]
            # run over axes array
            for i, x in enumerate(axes):
                graph_id = graph_ids[i]
                for j, ax in enumerate(x):
                    if j == 0:
                        draw_graph(graph_data, graph_id, ax, node_size=80, edge_width=2)
                        # set title on the left side of the plot
                        ax.set_ylabel(f"Graph {graph_id}\nLabel: {graph_data.graph_labels[graph_id]}")
                    else:
                        if i == 0:
                            ax.set_title(f"Convolution Layer {j}")
                        draw_graph_layer(graph_data, graph_id, layers[j-1], ax, node_size=20, edge_width=2)
            # draw_graph_layer(graph_data, graph_id, net.lr)
            # save the figure as svg
            plt.savefig(f'../Results/{db}/Figures/weights_run_{run}_val_step_{k_val}.svg')
            plt.show()






    except FileNotFoundError:
        print(f"Model {model_path} not found")
        return


if __name__ == "__main__":
    main()
