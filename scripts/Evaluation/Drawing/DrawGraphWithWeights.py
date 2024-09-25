import os
from pathlib import Path

import click
import matplotlib
import networkx as nx
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt

from scripts.find_best_models import load_preprocessed_data_and_parameters, config_paths_to_absolute
from src.Architectures.RuleGNN import RuleGNN
from src.utils.GraphData import GraphData, get_graph_data
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_splits import Load_Splits


def draw_graph(graph_data: GraphData, graph_id, ax, node_size=50, edge_color='black',
               edge_width=0.5, draw_type='circle'):
    '''
    Draw a graph with the given graph_id from the graph_data set
    '''
    graph = graph_data.graphs[graph_id]

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for node in graph.nodes():
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break

    node_labels = {}
    for node in graph.nodes():
        key = int(node)
        value = str(graph_data.node_labels['primary'].node_labels[graph_id][node])
        node_labels[key] = f'{value}'
    edge_labels = {}
    for (key1, key2, value) in graph.edges(data=True):
        if "label" in value and len(value["label"]) > 1:
            edge_labels[(key1, key2)] = int(value["label"][0])
        else:
            edge_labels[(key1, key2)] = ""
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_color, width=edge_width)
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='black')
    # get node colors from the node labels using the plasma colormap
    cmap = plt.get_cmap('tab20')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=graph_data.node_labels['primary'].num_unique_node_labels)
    node_colors = [cmap(norm(graph_data.node_labels['primary'].node_labels[graph_id][node])) for node in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=node_size)


def draw_graph_layer(results_path: Path, graph_data: GraphData, graph_id, layer, ax, cmap='seismic', node_size=50,
                     edge_width: float = 5.0, draw_type='circle', filter_weights=True, percentage=0.1, absolute=None,
                     with_graph=False):
    if with_graph:
        graph = graph_data.graphs[graph_id]

        # draw the graph
        # root node is the one with label 0
        root_node = None
        for node in graph.nodes():
            if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                root_node = node
                break

        # if graph is circular use the circular layout
        pos = dict()
        if draw_type == 'circle':
            # get circular positions around (0,0) starting with the root node at (-400,0)
            pos[root_node] = (400, 0)
            angle = 2 * np.pi / (graph.number_of_nodes())
            # iterate over the neighbors of the root node
            cur_node = root_node
            last_node = None
            counter = 0
            while len(pos) < graph.number_of_nodes():
                neighbors = list(graph.neighbors(cur_node))
                for next_node in neighbors:
                    if next_node != last_node:
                        counter += 1
                        pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                        last_node = cur_node
                        cur_node = next_node
                        break
        elif draw_type == 'kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.nx_pydot.graphviz_layout(graph)
        # keys to ints
        pos = {int(k): v for k, v in pos.items()}
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color='black', width=edge_width, alpha=0.5)

    all_weights = layer.get_weights()
    bias = layer.get_bias()
    graph = graph_data.graphs[graph_id]
    weight_distribution = layer.weight_distribution[graph_id]
    graph_weights = np.zeros_like(all_weights)
    for entry in weight_distribution:
        graph_weights[entry[2]] = all_weights[entry[2]]
    graph_weights = np.asarray(graph_weights)

    # sort weights
    if filter_weights:
        sorted_weights = np.sort(graph_weights)
        if absolute is None:
            lower_bound_weight = sorted_weights[int(len(sorted_weights) * percentage) - 1]
            upper_bound_weight = sorted_weights[int(len(sorted_weights) * (1 - percentage))]
        else:
            lower_bound_weight = sorted_weights[absolute - 1]
            upper_bound_weight = sorted_weights[-absolute]
        # set all weights smaller than the lower bound and larger than the upper bound to zero
        upper_weights = np.where(graph_weights >= upper_bound_weight, graph_weights, 0)
        lower_weights = np.where(graph_weights <= lower_bound_weight, graph_weights, 0)

        weights = upper_weights + lower_weights
    else:
        weights = np.asarray(graph_weights)
    bias_vector = np.asarray(bias)

    weight_min = np.min(graph_weights)
    weight_max = np.max(graph_weights)
    weight_max_abs = max(abs(weight_min), abs(weight_max))
    bias_min = np.min(bias_vector)
    bias_max = np.max(bias_vector)
    bias_max_abs = max(abs(bias_min), abs(bias_max))

    # use seismic colormap with maximum and minimum values from the weight matrix
    cmap = plt.get_cmap(cmap)
    # normalize item number values to colormap
    norm_weight = matplotlib.colors.Normalize(vmin=weight_min, vmax=weight_max)
    norm_bias = matplotlib.colors.Normalize(vmin=bias_min, vmax=bias_max)
    weight_colors = cmap(norm_weight(graph_weights))
    bias_colors = cmap(norm_bias(bias_vector))

    # draw the graph
    # root node is the one with label 0
    root_node = None
    for i, node in enumerate(graph.nodes()):
        if i == 0:
            print(f"First node: {graph_data.node_labels['primary'].node_labels[graph_id][node]}")
        if graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
            root_node = node
            break
    # if graph is circular use the circular layout
    pos = dict()
    if draw_type == 'circle':
        # get circular positions around (0,0) starting with the root node at (-400,0)
        pos[root_node] = (400, 0)
        angle = 2 * np.pi / (graph.number_of_nodes())
        # iterate over the neighbors of the root node
        cur_node = root_node
        last_node = None
        counter = 0
        while len(pos) < graph.number_of_nodes():
            neighbors = list(graph.neighbors(cur_node))
            for next_node in neighbors:
                if next_node != last_node:
                    counter += 1
                    pos[next_node] = (400 * np.cos(counter * angle), 400 * np.sin(counter * angle))
                    last_node = cur_node
                    cur_node = next_node
                    break
    elif draw_type == 'kawai':
        pos = nx.kamada_kawai_layout(graph)
    else:
        pos = nx.nx_pydot.graphviz_layout(graph)
    # keys to ints
    pos = {int(k): v for k, v in pos.items()}
    # graph to digraph with
    digraph = nx.DiGraph()
    for node in graph.nodes():
        digraph.add_node(node)

    edge_widths = []
    for entry in weight_distribution:
        i = entry[0]
        j = entry[1]
        if weights[entry[2]] != 0:
            # add edge with weight as data
            digraph.add_edge(i, j, weight=entry[2])
    curved_edges = [edge for edge in digraph.edges(data=True)]
    curved_edges_colors = []

    for edge in curved_edges:
        curved_edges_colors.append(weight_colors[edge[2]['weight']])
        edge_widths.append(edge_width * abs(weights[edge[2]['weight']]) / weight_max_abs)
    arc_rad = 0.25
    nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                           width=edge_widths,
                           connectionstyle=f'arc3, rad = {arc_rad}', arrows=True, arrowsize=5, node_size=5)

    node_colors = []
    node_sizes = []
    for node in digraph.nodes():
        node_label = graph_data.node_labels['primary'].node_labels[graph_id][node]
        node_colors.append(bias_colors[node_label])
        node_sizes.append(node_size * abs(bias_vector[node_label]) / bias_max_abs)

    nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)


@click.command()
@click.option('--db_name', default="DHFR", help='Database to use')
@click.option('--graph_ids', default="0", help='Graph ids to use')
@click.option('--config', default="", help='Path to the configuration file')
@click.option('--out', default="")
@click.option('--draw_type', default='circle')
# --data_path ../GraphBenchmarks/BenchmarkGraphs/ --db EvenOddRings2_16 --config ../TEMP/EvenOddRings2_16/config.yml
def main(db_name, graph_ids, config, out, draw_type, data_format='NEL'):
    # get the absolute path
    absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
    absolute_path = Path(absolute_path)
    run = 0
    validation_id = 0
    validation_folds = 10
    # load the model
    try:
        configs = yaml.safe_load(open(config))
    except FileNotFoundError:
        print(f"Config file {config} not found")
        return
    # get the data path from the config file
    config_paths_to_absolute(configs, absolute_path)
    results_path = absolute_path.joinpath(configs['paths']['results']).joinpath(db_name).joinpath('Results')
    if out == "":
        out = results_path
    else:
        out = Path(out)
    m_path = absolute_path.joinpath(configs['paths']['results']).joinpath(db_name).joinpath('Models')

    graph_data = get_graph_data(db_name=db_name, data_path=configs['paths']['data'], use_labels=configs['use_features'], use_attributes=configs['use_attributes'], graph_format=data_format)
    # adapt the precision of the input data
    if 'precision' in configs:
        if configs['precision'] == 'double':
            for i in range(len(graph_data.input_data)):
                graph_data.input_data[i] = graph_data.input_data[i].double()

    run_configs = get_run_configs(configs)

    for i, run_config in enumerate(run_configs):
        config_id = str(i).zfill(6)
        model_path = m_path.joinpath(f'model_Best_Configuration_{config_id}_run_{run}_val_step_{validation_id}.pt')
        seed = validation_id + validation_folds * run
        split_data = Load_Splits(absolute_path.joinpath(configs['paths']['splits']), db_name)
        test_data = np.asarray(split_data[0][validation_id], dtype=int)
        # check if the model exists
        try:
            with open(model_path, 'r'):
                para = Parameters()
                load_preprocessed_data_and_parameters(config_id=config_id, run_id=run, validation_id=validation_id,
                                                      graph_db_name=db_name, graph_data=graph_data,
                                                      run_config=run_config, para=para, validation_folds=validation_folds)

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)
                net = RuleGNN.RuleGNN(graph_data=graph_data,
                                      para=para,
                                      seed=seed, device=run_config.config.get('device', 'cpu'))

                net.load_state_dict(torch.load(model_path))
                # evaluate the performance of the model on the test data
                outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
                with torch.no_grad():
                    for j, data_pos in enumerate(test_data, 0):
                        inputs = torch.DoubleTensor(graph_data.input_data[data_pos])
                        outputs[j] = net(inputs, data_pos)
                    labels = graph_data.output_data[test_data]
                    # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                    counter = 0
                    correct = 0
                    for i, x in enumerate(outputs, 0):
                        if torch.argmax(x) == torch.argmax(labels[i]):
                            correct += 1
                        counter += 1
                    accuracy = correct / counter
                    print(f"Accuracy for model {model_path} is {accuracy}")
                # get the first three graphs from the test data
                graph_ids = test_data[[0, 20, 40]]
                #graph_ids = test_data[[0, 40, 80]]
                rows = len(graph_ids)
                cols = len(net.net_layers)
                with_filter = True
                if not with_filter:
                    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))
                    plt.subplots_adjust(wspace=0, hspace=0)
                    layers = net.net_layers
                    # run over axes array
                    for i, x in enumerate(axes):
                        graph_id = graph_ids[i]
                        for j, ax in enumerate(x):
                            if j == 0:
                                draw_graph(graph_data, graph_id, ax, node_size=40, edge_width=1, draw_type=draw_type)
                                # set title on the left side of the plot
                                ax.set_ylabel(f"Graph Label: ${graph_data.graph_labels[graph_id]}$")
                            else:
                                if i == 0:
                                    ax.set_title(f"Layer: ${j}$")
                                draw_graph_layer(results_path=results_path, graph_data=graph_data, graph_id=graph_id,
                                                 layer=layers[j - 1], ax=ax, node_size=40, edge_width=1,
                                                 draw_type=draw_type, filter_weights=False, percentage=1, absolute=None,
                                                 with_graph=True)
                else:
                    titles = ['All Weights', 'Top $10$ Weights', 'Top $3$ Weights']
                    filter_sizes = [None, 10, 3]
                    cols = len(filter_sizes) + 1
                    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))
                    plt.subplots_adjust(wspace=0, hspace=0)
                    layers = net.net_layers
                    # run over axes array
                    for i, x in enumerate(axes):
                        graph_id = graph_ids[i]
                        ax = x[0]
                        draw_graph(graph_data, graph_id, ax, node_size=20, edge_width=0.5, draw_type=draw_type)
                        # set title on the left side of the plot
                        ax.set_ylabel(f"Graph Label: ${graph_data.graph_labels[graph_id]}$")
                        for k, filter_size in enumerate(filter_sizes):
                            ax = x[k + 1]
                            if i == 0:
                                ax.set_title(titles[k])
                            draw_graph_layer(results_path=results_path, graph_data=graph_data, graph_id=graph_id,
                                             layer=layers[0], ax=ax, node_size=40, edge_width=1,
                                             draw_type=draw_type, filter_weights=True, percentage=1,
                                             absolute=filter_size, with_graph=True)

                # draw_graph_layer(graph_data, graph_id, net.lr)
                # save the figure as pdf using latex font
                plt.savefig(f'{out}/{db_name}_weights_run_{run}_val_step_{validation_id}.pdf', bbox_inches='tight', backend='pgf')
                #plt.savefig(f'{out}/{db}_weights_run_{run}_val_step_{validation_id}.svg', bbox_inches='tight')
                plt.show()






        except FileNotFoundError:
            print(f"Model {model_path} not found")
            return


if __name__ == "__main__":
    main()
