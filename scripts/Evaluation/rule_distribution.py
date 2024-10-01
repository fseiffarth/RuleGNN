### evaluate the distribution of rules in the datasets
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import GraphDrawing
from src.utils.GraphDrawing import CustomColorMap


def main():
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
    db_name = 'DHFR'

    # define nxm grid for the plots
    n = 3
    m = 5
    graph_id = 0
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
        aggregation_layer = net.net_layers[-1]
        convolution_layer.draw(ax=axs[run_id][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
        convolution_layer.draw(ax=axs[run_id][1], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 10})
        convolution_layer.draw(ax=axs[run_id][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 2})
        aggregation_layer.draw(ax=axs[run_id][3], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=0)
        aggregation_layer.draw(ax=axs[run_id][4], graph_id=graph_id, graph_drawing=graph_drawing, out_dimension=1)


    plt.show()


if __name__ == '__main__':
    main()
