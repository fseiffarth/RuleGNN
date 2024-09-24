
import os
from pathlib import Path
from typing import Tuple

import click
import matplotlib
import networkx as nx
import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
from pydot import Graph

from scripts.find_best_models import load_preprocessed_data_and_parameters, config_paths_to_absolute
from src.Architectures.RuleGNN import RuleGNN
from src.utils.GraphData import GraphData, get_graph_data
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_splits import Load_Splits

class GraphDrawing:
    def __init__(self, node_size=10.0, edge_width=1.0, weight_edge_width=1.0, weight_arrow_size=5.0, edge_color='black', edge_alpha=1, node_color='black', draw_type=None, colormap=plt.get_cmap('tab20')):
        self.node_size = node_size
        self.edge_width = edge_width
        self.weight_edge_width = weight_edge_width
        self.edge_color = edge_color
        self.edge_alpha = edge_alpha
        self.node_color = node_color
        self.arrow_size = weight_arrow_size
        self.draw_type = draw_type
        self.colormap = colormap

class WeightVisualization:
    def __init__(self, db_name, main_config, experiment_config, out="", data_format='NEL'):
        self.db_name = db_name
        try:
            main_config_datasets = yaml.load(open(main_config), Loader=yaml.FullLoader)
            for dataset in main_config_datasets['datasets']:
                if dataset['name'] == db_name:
                    self.main_config = dataset
                    break
        except FileNotFoundError:
            print(f"Main config file {main_config} not found")
            return
        try:
            self.experiment_config = yaml.load(open(experiment_config), Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"Experiment config file {experiment_config} not found")
            return
        self.out = out
        self.data_format = data_format
        # get the absolute path
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        absolute_path = Path(absolute_path)
        self.validation_folds = self.main_config['validation_folds']
        # get the data path from the config file
        config_paths_to_absolute(self.experiment_config, absolute_path)
        self.results_path = self.experiment_config['paths']['results'].joinpath(db_name).joinpath('Results')
        if out == "":
            self.out_path = self.experiment_config['paths']['results'].joinpath(db_name).joinpath('Plots')
        else:
            self.out_path = Path(out)
        self.m_path = self.experiment_config['paths']['results'].joinpath(db_name).joinpath('Models')

        self.graph_data = get_graph_data(db_name=db_name,
                                         data_path=self.experiment_config['paths']['data'],
                                         input_features=self.experiment_config.get('input_features', None),
                                         graph_format=data_format)



    def draw_graph(self, graph_id, ax, graph_drawing: Tuple[GraphDrawing, GraphDrawing]):
        '''
        Draw a graph with the given graph_id from the graph_data set
        '''
        graph = self.graph_data.graphs[graph_id]

        # draw the graph
        # root node is the one with label 0
        root_node = None
        for node in graph.nodes():
            if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                root_node = node
                break

        node_labels = {}
        for node in graph.nodes():
            key = int(node)
            value = str(self.graph_data.node_labels['primary'].node_labels[graph_id][node])
            node_labels[key] = f'{value}'
        edge_labels = {}
        for (key1, key2, value) in graph.edges(data=True):
            if "label" in value and len(value["label"]) > 1:
                edge_labels[(key1, key2)] = int(value["label"][0])
            else:
                edge_labels[(key1, key2)] = ""
        # if graph is circular use the circular layout
        pos = dict()
        if graph_drawing[0].draw_type == 'circle':
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
        elif graph_drawing[0].draw_type == 'kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.nx_pydot.graphviz_layout(graph)
        # keys to ints
        pos = {int(k): v for k, v in pos.items()}
        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[0].edge_color, width=graph_drawing[0].edge_width)
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color='black')
        # get node colors from the node labels using the plasma colormap
        cmap = graph_drawing[0].colormap
        norm = matplotlib.colors.Normalize(vmin=0, vmax=self.graph_data.node_labels['primary'].num_unique_node_labels)
        node_colors = [cmap(norm(self.graph_data.node_labels['primary'].node_labels[graph_id][node])) for node in graph.nodes()]
        nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_color=node_colors, node_size=graph_drawing[0].node_size)


    def draw_graph_layer(self, graph_id, layer, ax, graph_drawing: Tuple[GraphDrawing, GraphDrawing], filter_weights=True, percentage=0.1, absolute=None,
                         with_graph=False):
        if with_graph:
            graph = self.graph_data.graphs[graph_id]

            # draw the graph
            # root node is the one with label 0
            root_node = None
            for node in graph.nodes():
                if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                    root_node = node
                    break

            # if graph is circular use the circular layout
            pos = dict()
            if graph_drawing[0].draw_type == 'circle':
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
            elif graph_drawing[0].draw_type == 'kawai':
                pos = nx.kamada_kawai_layout(graph)
            else:
                pos = nx.nx_pydot.graphviz_layout(graph)
            # keys to ints
            pos = {int(k): v for k, v in pos.items()}
            nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=graph_drawing[1].edge_color, width=graph_drawing[1].edge_width, alpha=graph_drawing[1].edge_alpha*0.5)

        all_weights = layer.get_weights()
        bias = layer.get_bias()
        graph = self.graph_data.graphs[graph_id]
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
        cmap = graph_drawing[1].colormap
        # normalize item number values to colormap
        norm_weight = matplotlib.colors.Normalize(vmin=weight_min, vmax=weight_max)
        norm_bias = matplotlib.colors.Normalize(vmin=bias_min, vmax=bias_max)
        normed_weight = (graph_weights + (-weight_min)) / (weight_max - weight_min)
        weight_colors = cmap(normed_weight)
        normed_bias = (bias_vector + (-bias_min)) / (bias_max - bias_min)
        bias_colors = cmap(normed_bias)

        # draw the graph
        # root node is the one with label 0
        root_node = None
        for i, node in enumerate(graph.nodes()):
            if i == 0:
                print(f"First node: {self.graph_data.node_labels['primary'].node_labels[graph_id][node]}")
            if self.graph_data.node_labels['primary'].node_labels[graph_id][node] == 0:
                root_node = node
                break
        # if graph is circular use the circular layout
        pos = dict()
        if graph_drawing[0].draw_type == 'circle':
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
        elif graph_drawing[0].draw_type == 'kawai':
            pos = nx.kamada_kawai_layout(graph)
        else:
            pos = nx.nx_pydot.graphviz_layout(graph)
        # keys to ints
        pos = {int(k): v for k, v in pos.items()}
        # graph to digraph with
        digraph = nx.DiGraph()
        for node in graph.nodes():
            digraph.add_node(node)



        node_colors = []
        node_sizes = []
        for node in digraph.nodes():
            node_label = layer.node_labels.node_labels[graph_id][node]
            node_colors.append(bias_colors[node_label])
            node_sizes.append(graph_drawing[1].node_size * abs(bias_vector[node_label]) / bias_max_abs)

        nx.draw_networkx_nodes(digraph, pos=pos, ax=ax, node_color=node_colors, node_size=node_sizes)

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
            edge_widths.append(graph_drawing[1].weight_edge_width * abs(weights[edge[2]['weight']]) / weight_max_abs)
        arc_rad = 0.25
        nx.draw_networkx_edges(digraph, pos, ax=ax, edgelist=curved_edges, edge_color=curved_edges_colors,
                               width=edge_widths,
                               connectionstyle=f'arc3, rad = {arc_rad}', arrows=True, arrowsize=graph_drawing[1].arrow_size, node_size=graph_drawing[1].node_size)




    def visualize(self, graph_ids, run=0, validation_id=0, graph_drawing: Tuple[GraphDrawing, GraphDrawing] = None, filter_sizes = (None, 10, 3)):
        # adapt the precision of the input data
        if self.experiment_config.get('precision', 'double') == 'double':
            for i in range(len(self.graph_data.inputs)):
                self.graph_data.inputs[i] = self.graph_data.inputs[i].double()


        run_configs = get_run_configs(self.experiment_config)

        if graph_drawing is None:
            graph_drawing = (GraphDrawing(node_size=10, edge_width=0.5), GraphDrawing(node_size=10, edge_width=3))

        for i, run_config in enumerate(run_configs):
            config_id = str(i).zfill(6)
            model_path = self.m_path.joinpath(f'model_Best_Configuration_{config_id}_run_{run}_val_step_{validation_id}.pt')
            # check if the model exists
            if model_path.exists():
                with open(model_path, 'r'):
                    seed = validation_id + self.validation_folds * run
                    split_data = Load_Splits(self.experiment_config['paths']['splits'], self.db_name)
                    test_data = np.asarray(split_data[0][validation_id], dtype=int)
                    graph_ids = test_data[graph_ids]
                    para = Parameters()
                    load_preprocessed_data_and_parameters(config_id=config_id, run_id=run, validation_id=validation_id,
                                                          graph_db_name=self.db_name, graph_data=self.graph_data,
                                                          run_config=run_config, para=para, validation_folds=self.validation_folds)

                    """
                        Get the first index in the results directory that is not used
                    """
                    para.set_file_index(size=6)
                    net = RuleGNN.RuleGNN(graph_data=self.graph_data,
                                          para=para,
                                          seed=seed, device=run_config.config.get('device', 'cpu'))

                    net.load_state_dict(torch.load(model_path))
                    # evaluate the performance of the model on the test data
                    outputs = torch.zeros((len(test_data), self.graph_data.num_classes), dtype=torch.double)
                    with torch.no_grad():
                        for j, data_pos in enumerate(test_data, 0):
                            inputs = torch.DoubleTensor(self.graph_data.inputs[data_pos])
                            outputs[j] = net(inputs, data_pos)
                        labels = self.graph_data.one_hot_labels[test_data]
                        # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                        counter = 0
                        correct = 0
                        for i, x in enumerate(outputs, 0):
                            if torch.argmax(x) == torch.argmax(labels[i]):
                                correct += 1
                            counter += 1
                        accuracy = correct / counter
                        print(f"Accuracy for model {model_path} is {accuracy}")
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
                                    self.draw_graph(graph_id, ax, graph_drawing)
                                    # set title on the left side of the plot
                                    ax.set_ylabel(f"Graph Label: ${self.graph_data.graph_labels[graph_id]}$")
                                else:
                                    if i == 0:
                                        ax.set_title(f"Layer: ${j}$")
                                    self.draw_graph_layer(graph_id=graph_id,
                                                     layer=layers[j - 1], ax=ax, graph_drawing=graph_drawing, filter_weights=False, percentage=1, absolute=None,
                                                     with_graph=True)
                    else:
                        titles = ['All Weights', 'Top $10$ Weights', 'Top $3$ Weights']

                        cols = len(filter_sizes) + 1
                        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))
                        plt.subplots_adjust(wspace=0, hspace=0)
                        layers = net.net_layers
                        # run over axes array
                        for i, x in enumerate(axes):
                            graph_id = graph_ids[i]
                            ax = x[0]
                            self.draw_graph(graph_id, ax, graph_drawing=graph_drawing)
                            # set title on the left side of the plot
                            ax.set_ylabel(f"Graph Label: ${self.graph_data.graph_labels[graph_id]}$")
                            for k, filter_size in enumerate(filter_sizes):
                                ax = x[k + 1]
                                if i == 0:
                                    ax.set_title(titles[k])
                                self.draw_graph_layer(graph_id=graph_id,
                                                 layer=layers[0], ax=ax, graph_drawing=graph_drawing, filter_weights=True, percentage=1,
                                                 absolute=filter_size, with_graph=True)

                    # draw_graph_layer(self.graph_data, graph_id, net.lr)
                    # save the figure as pdf using latex font
                    save_path = self.out_path.joinpath(f'{self.db_name}_weights_run_{run}_val_step_{validation_id}.pdf')
                    plt.savefig(save_path, bbox_inches='tight', backend='pgf', dpi=600)
                    #plt.show()
            else:
                print(f"Model {model_path} not found")

