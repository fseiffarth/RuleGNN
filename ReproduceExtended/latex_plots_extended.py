from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Reproduce.latex import share_gnn_results
from src.Experiment.ExperimentMain import ExperimentMain
from src.Architectures.ShareGNN.ShareGNNLayers import InvariantBasedMessagePassingLayer
from src.utils.GraphDrawing import GraphDrawing, CustomColorMap, RandomColorMap

def plot_network(path, db_name, graph_ids, filtering, draw_type=None, with_labels_from_invariant=True, with_aggregation=False, molecule=False, channel=0):
    graph_id_string = '_'.join([str(graph_id) for graph_id in graph_ids])
    # check if file exists
    if not Path(f'Reproduce/Results/Latex/Plots/visualization_{db_name}_{graph_id_string}.pdf').exists():
        #mpl.use("pgf")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 18,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })

        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False
        #experiment = ExperimentMain(Path('Reproduce/Configs/main_config_fair_real_world.yml'))
        #experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
        experiment = ExperimentMain(Path(path))

        net = experiment.load_model(db_name=db_name, run_id=0, validation_id=0, best=True)
        num_convolution_layers = 0
        for layers in net.net_layers:
            if isinstance(layers, InvariantBasedMessagePassingLayer):
                num_convolution_layers += 1
        #sort_indices, steps = rules_vs_occurences(convolution_layer, db_name, channel)
        #rules_vs_occurences_properties(convolution_layer)
        #rules_vs_weights(convolution_layer, sort_indices, steps, db_name, channel)

        n = len(graph_ids)
        column_for_invariants = 1 if with_labels_from_invariant else 0
        m = 1 + len(filtering)*num_convolution_layers + column_for_invariants

        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 5*n))
        plt.subplots_adjust(wspace=0, hspace=0)
        graph_drawing = (
            GraphDrawing(node_size=160, edge_width=1, draw_type=draw_type),
            GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                         colormap=CustomColorMap().cmap, draw_type=draw_type)
        )
        # use plasma colormap for the bias
        graph_bias_drawing = (
            GraphDrawing(node_size=160, edge_width=1, colormap=RandomColorMap('nipy_spectral', 99999).cmap, draw_type=draw_type),
            GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10, draw_type=draw_type)
        )

        Path('Reproduce/Results/Latex/Plots/Positions/').mkdir(exist_ok=True, parents=True)
        save_pos_path = Path('Reproduce/Results/Latex/Plots/Positions/')

        if len(graph_ids) == 1:
            pos_path = save_pos_path.joinpath(f'{db_name}_{graph_ids[0]}_pos.txt')
            convolution_layer = net.net_layers[0]
            convolution_layer.draw(ax=axs[0], graph_id=graph_ids[0], graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
            if with_labels_from_invariant:
                convolution_layer.draw(ax=axs[1], graph_id=graph_ids[0], graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)

            for i in range(0, num_convolution_layers):
                # get convolution layer
                convolution_layer = net.net_layers[i]
                for filter_weights in filtering:
                    convolution_layer.draw(ax=axs[2+i], graph_id=graph_ids[0], graph_drawing=graph_drawing, filter_weights=filter_weights, pos_path=pos_path)

            # add subplots column and row titles
            if molecule:
                axs[0].set_title(f'Atomic Numbers')
            else:
                axs[0].set_title(f'Node Labels')
            if with_labels_from_invariant:
                axs[1].set_title(f'Labels from Invariant')
            for i in range(0, num_convolution_layers):
                if num_convolution_layers > 1:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Layer {i+1}')
                        elif 'absolute' in filter_weights:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Layer {i+1} Top ${filter_weights["absolute"]}$ Weights')
                else:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'All Weights')
                        elif 'absolute' in filter_weights:
                            axs[1+column_for_invariants+ i*len(filtering) + j].set_title(f'Top ${filter_weights["absolute"]}$ Weights')

            axs[0].set_ylabel(f'Graph Label: {net.graph_data.y[graph_ids[0]].item()}')
        else:
            for idx, graph_id in enumerate(graph_ids):
                pos_path = save_pos_path.joinpath(f'{db_name}_{graph_id}_pos.txt')

                # get convolution layer
                convolution_layer = net.net_layers[0]
                convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
                if with_labels_from_invariant:
                    convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)
                for i in range(0, num_convolution_layers):
                    # get convolution layer
                    convolution_layer = net.net_layers[i]
                    for j, filter_weights in enumerate(filtering):
                        convolution_layer.draw(ax=axs[idx][1+column_for_invariants+ i*len(filtering) + j], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=filter_weights, pos_path=pos_path)

            # add subplots column and row titles
            # add subplots column and row titles
            if molecule:
                axs[0][0].set_title(f'Atomic Numbers')
            else:
                axs[0][0].set_title(f'Node Labels')
            if with_labels_from_invariant:
                axs[0][1].set_title(f'Labels from Invariant')
            for i in range(0, num_convolution_layers):
                if num_convolution_layers > 1:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Layer {i+1}')
                        elif 'absolute' in filter_weights:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Layer {i+1} Top ${filter_weights["absolute"]}$ Weights')
                else:
                    for j, filter_weights in enumerate(filtering):
                        if filter_weights is None:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'All Weights')
                        elif 'absolute' in filter_weights:
                            axs[0][1 + column_for_invariants + i*len(filtering) + j].set_title(f'Top ${filter_weights["absolute"]}$ Weights')

            for idx, graph_id in enumerate(graph_ids):
                axs[idx][0].set_ylabel(f'Graph Label: ${net.graph_data.y[graph_id].item()}$')


        plt.savefig(f'Reproduce/Results/Latex/Plots/visualization_{db_name}_{graph_id_string}.pdf', bbox_inches='tight', backend='pgf')
        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True


def plot_specific_graphs_from_db(path, db_name, graph_ids, draw_type=None, node_size=200, output_path=None):
    if output_path is None:
        output_path = Path(f'Reproduce/Results/Latex/Plots/')
    else:
        output_path = Path(output_path)
    if not output_path.joinpath(f'{db_name}_{"_".join(map(str, graph_ids))}.pdf').exists():
        # make dir f'scripts/Evaluation/Drawing/Graphs/{db_name}/' if it does not exist
        output_path.mkdir(exist_ok=True, parents=True)

        #mpl.use("pgf")
        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "font.size": 12,
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.texsystem": "lualatex",
            "pgf.preamble": "\n".join([
                r"\usepackage{url}",  # load additional packages
                r"\usepackage{unicode-math}",  # unicode math setup
                r"\setmainfont{DejaVu Serif}",  # serif font via preamble
            ])
        })
        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False

        experiment = ExperimentMain(Path(path))
        net = experiment.load_model(db_name=db_name, run_id=0, validation_id=0, best=True)

        fig, ax = plt.subplots(1, len(graph_ids), figsize=(5*len(graph_ids), 5*1))
        for i, graph_id in enumerate(graph_ids):
            plt.subplots_adjust(wspace=0, hspace=0)
            graph_drawing = (
                GraphDrawing(node_size=node_size, edge_width=1, draw_type=draw_type),
                GraphDrawing(node_size=node_size, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                             colormap=CustomColorMap().cmap, draw_type=draw_type)
            )
            # get convolution layer
            convolution_layer = net.net_layers[0]
            convolution_layer.draw(ax=ax[i], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)

            # add subplots column and row titles
            #axs.set_title(f'Graphs with Atomic Numbers')
            ax[i].set_xlabel(f'Graph Label: ${net.graph_data.y[graph_id]}$')



        plt.savefig(output_path.joinpath(f'{db_name}_{"_".join(map(str, graph_ids))}.pdf', bbox_inches='tight', backend='pgf'))

        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = True
        plt.rcParams['axes.spines.right'] = True
        plt.rcParams['axes.spines.top'] = True
        plt.rcParams['axes.spines.bottom'] = True


def main():
    # create Latex dir under Results
    Path('ReproduceExtended/Results/Latex').mkdir(parents=True, exist_ok=True)
    Path('ReproduceExtended/Results/Latex/Plots').mkdir(parents=True, exist_ok=True)
    plot_network_path = 'ReproduceExtended/configs/main_config_substructure_counting.yml'
    plot_network_path_random = 'Reproduce/Configs/main_config_fair_real_world_random_variation.yml'
    plot_network_path_synthetic = 'Reproduce/Configs/main_config_fair_synthetic.yml'

    plot_network_path_ablation_threshold = lambda x : f'Reproduce/Configs/ablation/threshold/lower/main_config_ablation_threshold_{x}.yml'
    plot_network_path_ablation_distance = 'Reproduce/Configs/ablation/distances/main_config_ablation_distances.yml'




    plot_network(plot_network_path_synthetic, 'EvenOddRings2_16', [2, 1, 4], with_labels_from_invariant=False, draw_type='circle', filtering=[None, {'absolute' : 5}])
    plot_network(plot_network_path_synthetic, 'EvenOddRingsCount16', [0,5,6], with_labels_from_invariant=False, draw_type='circle', filtering=[None])
    plot_network(plot_network_path_synthetic, 'Snowflakes', [500,120,476], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path_synthetic, 'CSL', [0,16,31], draw_type='kawai', filtering=[None])
    plot_network(plot_network_path_random, 'IMDB-MULTI', [25,805,1265], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path_random, 'IMDB-BINARY', [101,68,612], draw_type='kawai', filtering=[None, {'absolute' : 3}])

    plot_network(plot_network_path_random, 'DHFR', [272, 273], draw_type='kawai', filtering=[None, {'absolute' : 3}], molecule=True)
    plot_network(plot_network_path, 'NCI1', [216, 320, 655], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path, 'NCI109', [56, 18, 3165], draw_type='kawai', filtering=[None, {'absolute' : 3}])
    plot_network(plot_network_path, 'Mutagenicity', [1654, 257, 360], draw_type='kawai', filtering=[None, {'absolute' : 3}])








if __name__ == '__main__':
    main()