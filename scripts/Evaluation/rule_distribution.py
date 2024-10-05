### evaluate the distribution of rules in the datasets
from pathlib import Path

from matplotlib import pyplot as plt

from scripts.Evaluation.Drawing.plotting import rules_vs_occurences, rules_vs_weights, rules_vs_occurences_properties
from scripts.ExperimentMain import ExperimentMain
from scripts.WeightVisualization import GraphDrawing
from src.utils.GraphDrawing import CustomColorMap


def main():
    #experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_real_world.yml'))
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
    db_name = 'DHFR'

    net = experiment.load_model(db_name=db_name, config_id=0, run_id=0, validation_id=0)
    convolution_layer = net.net_layers[-2]
    sort_indices = rules_vs_occurences(convolution_layer)
    #rules_vs_occurences_properties(convolution_layer)
    rules_vs_weights(convolution_layer, sort_indices)
    return
    # define nxm grid for the plots
    n = 3
    m = 8
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
