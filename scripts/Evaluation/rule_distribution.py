### evaluate the distribution of rules in the datasets
from pathlib import Path

from matplotlib import pyplot as plt
from scripts.Evaluation.Drawing.plotting import rules_vs_occurences, rules_vs_weights
from scripts.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import GraphDrawing
from src.utils.GraphDrawing import CustomColorMap, TabColorMap


def main():
    import matplotlib as mpl

    #mpl.use("pgf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "text.usetex": True,  # use inline math for ticks
        "pgf.rcfonts": False,  # don't setup fonts from rc parameters
        "pgf.texsystem": "lualatex",
        "pgf.preamble": "\n".join([
            r"\usepackage{url}",  # load additional packages
            r"\usepackage{unicode-math}",  # unicode math setup
            r"\setmainfont{DejaVu Serif}",  # serif font via preamble
        ])
    })
    #experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_real_world.yml'))
    #experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))


    experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_real_world.yml'))
    db_name = 'DHFR'
    #experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_synthetic.yml'))
    #db_name = 'LongRings100'

    net = experiment.load_model(db_name=db_name, config_id=41, run_id=0, validation_id=0)
    convolution_layer = net.net_layers[-2]
    channel = 0
    #sort_indices, steps = rules_vs_occurences(convolution_layer, db_name, channel)
    #rules_vs_occurences_properties(convolution_layer)
    #rules_vs_weights(convolution_layer, sort_indices, steps, db_name, channel)
    # define nxm grid for the plots
    n = 3
    m = 4

    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 5*n))
    plt.subplots_adjust(wspace=0, hspace=0)
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                     colormap=CustomColorMap().cmap)
    )
    # use plasma colormap for the bias
    graph_bias_drawing = (
        GraphDrawing(node_size=40, edge_width=1, colormap=TabColorMap().cmap),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10)
    )
    # for idx, graph_id in enumerate([0,5,4]):
    #     net = experiment.load_model(db_name=db_name, config_id=0, run_id=0, validation_id=0)
    #     # get convolution layer
    #     convolution_layer = net.net_layers[0]
    #     aggregation_layer = net.net_layers[-1]
    #     convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
    #     convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None)
    #     convolution_layer.draw(ax=axs[idx][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 10})
    #     convolution_layer.draw(ax=axs[idx][3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
    #     aggregation_layer.draw(ax=axs[idx][4], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=0)
    #     aggregation_layer.draw(ax=axs[idx][5], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=1)

    # # add subplots column and row titles
    # axs[0][0].set_title(f'Graphs')
    # axs[0][1].set_title(f'Attention Coefficients')
    # axs[0][2].set_title(f'Top $10$ Attention Coefficients')
    # axs[0][3].set_title(f'Top $3$ Attention Coefficients')
    # axs[0][4].set_title(f'Output Neuron $1$ Activations')
    # axs[0][5].set_title(f'Output Neuron $2$ Activations')

    #for idx, graph_id in enumerate([272,273,274]):
    for idx, graph_id in enumerate([746, 747, 748]):
        # get convolution layer
        convolution_layer = net.net_layers[0]
        aggregation_layer = net.net_layers[-1]
        convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
        convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True)
        convolution_layer.draw(ax=axs[idx][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None)
        convolution_layer.draw(ax=axs[idx][3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})

    # add subplots column and row titles
    axs[0][0].set_title(f'Graphs with Atom Labels')
    axs[0][1].set_title(f'Graphs with Rule Labels')
    axs[0][2].set_title(f'All Coefficients')
    axs[0][3].set_title(f'Top $5$ Coefficients')

    for idx, graph_id in enumerate([0,5,4]):
        axs[idx][0].set_ylabel(f'Graph Label {net.graph_data.graph_labels[graph_id]}')

    plt.savefig(f'scripts/Evaluation/Drawing/Figures/visualization_{db_name}.png')
    plt.show()
    return


    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(3*n, 3*m))
    plt.subplots_adjust(wspace=0, hspace=0)
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                     colormap=CustomColorMap().cmap)
    )

    for run_id in range(3):
        net = experiment.load_model(db_name=db_name, config_id=0, run_id=run_id, validation_id=0)
        # get convolution layer
        convolution_layer = net.net_layers[0]
        convolution_layer1 = net.net_layers[1]
        convolution_layer2 = net.net_layers[2]
        convolution_layer3 = net.net_layers[3]
        convolution_layer4 = net.net_layers[4]
        aggregation_layer = net.net_layers[-1]
        convolution_layer.draw(ax=axs[run_id][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
        convolution_layer.draw(ax=axs[run_id][1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
        convolution_layer1.draw(ax=axs[run_id][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
        convolution_layer2.draw(ax=axs[run_id][3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
        convolution_layer3.draw(ax=axs[run_id][4], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
        convolution_layer4.draw(ax=axs[run_id][5], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3})
        aggregation_layer.draw(ax=axs[run_id][6], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=0)
        aggregation_layer.draw(ax=axs[run_id][7], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=1)






    plt.show()


if __name__ == '__main__':
    main()
