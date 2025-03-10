### evaluate the distribution of rules in the datasets
from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import GraphDrawing
from src.utils.GraphDrawing import CustomColorMap


def plot_all_graphs_from_db(db_name, experiment):
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
    #experiment = ExperimentMain(Path('Reproduce/Configs/main_config_fair_real_world.yml'))
    #experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))



    net = experiment.load_model(db_name=db_name, config_id=41, run_id=0, validation_id=0)

    # make dir f'scripts/Evaluation/Drawing/Graphs/{db_name}/' if it does not exist
    Path(f'scripts/Evaluation/Drawing/Graphs/{db_name}').mkdir(exist_ok=True, parents=True)

    for graph_id in range(len(net.graph_data)):
        # check if the graph is already in the dataset
        if Path(f'scripts/Evaluation/Drawing/Graphs/{db_name}/{db_name}_{graph_id}.png').exists():
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
        # remove matplotlib frame
        # remove frame from each side of plot
        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.right'] = False
        plt.rcParams['axes.spines.top'] = False
        plt.rcParams['axes.spines.bottom'] = False
        #axs.set_title(f'Graphs with Atom Labels')
        axs.set_xlabel(f'Graph Label {net.graph_data.y[graph_id]}')



        plt.savefig(f'scripts/Evaluation/Drawing/Graphs/{db_name}/{db_name}_{graph_id}.png', bbox_inches='tight')
        plt.close()




if __name__ == '__main__':
    experiment = ExperimentMain(Path('Reproduce/Configs/main_config_fair_real_world.yml'))
    experiment = ExperimentMain(Path('Reproduce/Configs/main_config_fair_synthetic.yml'))
    plot_all_graphs_from_db(db_name='CSL', experiment=experiment)
    #plot_all_graphs_from_db(db_name='EvenOddRingsCount16', experiment=experiment)
    #plot_all_graphs_from_db(db_name='EvenOddRings2_16', experiment=experiment)
    #plot_all_graphs_from_db(db_name='Snowflakes', experiment=experiment)
    plot_specific_graphs_from_db(db_name='EvenOddRings2_16', experiment=experiment, graph_ids=[3,2,1,4])
    plot_specific_graphs_from_db(db_name='EvenOddRingsCount16', experiment=experiment, graph_ids=[0,5])
    plot_specific_graphs_from_db(db_name='LongRings100', experiment=experiment, graph_ids=[1,3,5])
