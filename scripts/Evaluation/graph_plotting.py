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
    net = experiment.load_model(db_name=db_name, config_id=41, run_id=0, validation_id=0)
    for graph_id in range(net.graph_data.num_graphs):
        # check if the graph is already in the dataset
        if Path(f'scripts/Evaluation/Drawing/Graphs/DHFR/dhfr_{graph_id}.png').exists():
            continue
        n = 1
        m = 1

        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5*m, 5*n))
        plt.subplots_adjust(wspace=0, hspace=0)
        graph_drawing = (
            GraphDrawing(node_size=40, edge_width=1),
            GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,
                         colormap=CustomColorMap().cmap)
        )
        # get convolution layer
        convolution_layer = net.net_layers[0]
        convolution_layer.draw(ax=axs, graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)

        # add subplots column and row titles
        axs.set_title(f'Graphs with Atom Labels')
        axs.set_ylabel(f'Graph Label {net.graph_data.graph_labels[graph_id]}')


        plt.savefig(f'scripts/Evaluation/Drawing/Graphs/DHFR/dhfr_{graph_id}.png')
        plt.close()


if __name__ == '__main__':
    main()
