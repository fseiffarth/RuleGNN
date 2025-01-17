### evaluate the distribution of rules in the datasets
from pathlib import Path

from matplotlib import pyplot as plt
from scripts.Evaluation.Drawing.plotting import rules_vs_occurences, rules_vs_weights
from scripts.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import GraphDrawing
from src.utils.GraphDrawing import CustomColorMap, TabColorMap, RandomColorMap


def main():
    import matplotlib as mpl

    #mpl.use("pgf")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",  # use serif/main font for text elements
        "font.size": 24,
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


    experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_real_world_random_variation.yml'))
    db_name = 'DHFR'

    net = experiment.load_model(db_name=db_name, run_id=0, validation_id=0, best=True)
    convolution_layer = net.net_layers[-2]
    channel = 0
    #sort_indices, steps = rules_vs_occurences(convolution_layer, db_name, channel)
    #rules_vs_occurences_properties(convolution_layer)
    #rules_vs_weights(convolution_layer, sort_indices, steps, db_name, channel)
    # define nxm grid for the plots
    graph_ids = [746, 747, 748]
    graph_ids = [272, 273, 274]
    graph_ids = [272, 273]
    n = len(graph_ids)
    m = 4

    fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 5*n))
    plt.subplots_adjust(wspace=0, hspace=0)
    graph_drawing = (
        GraphDrawing(node_size=160, edge_width=1),
        GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                     colormap=CustomColorMap().cmap)
    )
    # use plasma colormap for the bias
    graph_bias_drawing = (
        GraphDrawing(node_size=160, edge_width=1, colormap=RandomColorMap('nipy_spectral', 99999).cmap),
        GraphDrawing(node_size=160, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10)
    )

    save_pos_path = Path('Reproduce_RuleGNN/Results/Drawing/')

    if len(graph_ids) == 1:
        pos_path = save_pos_path.joinpath(f'{db_name}_{graph_ids[0]}_pos.txt')
        # get convolution layer
        convolution_layer = net.net_layers[0]
        aggregation_layer = net.net_layers[-1]
        convolution_layer.draw(ax=axs[0], graph_id=graph_ids[0], graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
        convolution_layer.draw(ax=axs[1], graph_id=graph_ids[0], graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)
        convolution_layer.draw(ax=axs[2], graph_id=graph_ids[0], graph_drawing=graph_drawing, filter_weights=None, pos_path=pos_path)
        convolution_layer.draw(ax=axs[3], graph_id=graph_ids[0], graph_drawing=graph_drawing, filter_weights={'absolute': 3}, pos_path=pos_path)

        # add subplots column and row titles
        axs[0].set_title(f'Atom Labels')
        axs[1].set_title(f'Labels from Invariant')
        axs[2].set_title(f'Learned Parameters')
        axs[3].set_title(f'Top $3$ Parameters')

        axs[0].set_ylabel(f'Graph Label: {net.graph_data.y[graph_ids[0]].item()}')
    else:
        for idx, graph_id in enumerate(graph_ids):
            pos_path = save_pos_path.joinpath(f'{db_name}_{graph_id}_pos.txt')
            # get convolution layer
            convolution_layer = net.net_layers[0]
            aggregation_layer = net.net_layers[-1]
            convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True, pos_path=pos_path)
            convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True, pos_path=pos_path)
            convolution_layer.draw(ax=axs[idx][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None, pos_path=pos_path)
            convolution_layer.draw(ax=axs[idx][3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3}, pos_path=pos_path)

        # add subplots column and row titles
        axs[0][0].set_title(f'Atom Labels')
        axs[0][1].set_title(f'Labels from Invariant')
        axs[0][2].set_title(f'Learned Parameters')
        axs[0][3].set_title(f'Top $3$ Parameters')

        for idx, graph_id in enumerate(graph_ids):
            axs[idx][0].set_ylabel(f'Graph Label: {net.graph_data.y[graph_id].item()}')

    plt.savefig(f'Reproduce_RuleGNN/Results/Drawing/visualization_{db_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()