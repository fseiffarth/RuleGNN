from pathlib import Path

import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import numpy as np

from src.Experiment.ExperimentMain import ExperimentMain
from src.utils.GraphDrawing import GraphDrawing
from src.utils.load_splits import Load_Splits


class CustomColorMap:
    def __init__(self):
        aqua = (0.0, 0.6196, 0.8902)
        # 89,189,247
        skyblue = (0.3490, 0.7412, 0.9686)
        fuchsia = (232 / 255.0, 46 / 255.0, 130 / 255.0)
        violet = (152 / 255.0, 48 / 255.0, 130 / 255.0)
        white = (1.0, 1.0, 1.0)
        # darknavy 12,18,43
        darknavy = (12 / 255.0, 18 / 255.0, 43 / 255.0)

        # Define the three colors and their positions
        lamarr_colors = [aqua, white, fuchsia] # Color 3 (RGB values)

        positions = [0.0, 0.5, 1.0]  # Positions of the colors (range: 0.0 to 1.0)

        # Create a colormap using LinearSegmentedColormap
        self.cmap = mcolors.LinearSegmentedColormap.from_list('custom_colormap', list(zip(positions, lamarr_colors)))


def get_graph_text(graph_id, data, text_column='Text'):
    return data.iloc[graph_id][text_column]

def get_graph_label(graph_id, data, label_column='Label'):
    return data.iloc[graph_id][label_column]

def get_graph_tokens(graph_id, graph_data, token_list, text_column='Text'):
    primary_labels = graph_data.node_labels['primary'].node_labels[graph_data.slices['x'][graph_id]:graph_data.slices['x'][graph_id + 1]]
    return [token_list[primary_label] for primary_label in primary_labels]


def main():
    experiment = ExperimentMain(Path('Examples/TextClassification/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    for db_name in ['sentiment_small']:

        validation_id = 2
        configuration = experiment.network_configurations[db_name][0]
        path_to_data = Path(configuration['paths']['data']) / f'{db_name}'
        with open(path_to_data / f'{db_name}_plain_text.txt', 'r') as f:
            plain_texts = [line.strip() for line in f.readlines()]
        with open(path_to_data / f'{db_name}_tokenized_text.txt', 'r') as f:
            tokenized_texts = [line.strip() for line in f.readlines()]
        with open(path_to_data / f'{db_name}_labels.txt', 'r') as f:
            output_labels = [line.strip() for line in f.readlines()]
        split_data = Load_Splits(configuration['paths']['splits'], db_name)
        test_data = np.asarray(split_data[0][validation_id], dtype=int)
        graph_ids = test_data
        net = experiment.load_model(db_name=db_name, config_id=0, run_id=0, validation_id=2, best=False)
        outputs, labels, accuracy = experiment.evaluate_model_on_graphs(db_name=db_name, graph_ids=graph_ids, config_id=0, run_id=0, validation_id=2, best=False)
        arg_max_outputs = np.argmax(outputs, axis=1)
        correct_outputs = np.equal(arg_max_outputs, labels)
        n = len(graph_ids)
        m = 6

        fig, axs = plt.subplots(nrows=n, ncols=m, figsize=(5 * m, 5 * n))
        plt.subplots_adjust(wspace=0, hspace=0)
        graph_drawing = (
            GraphDrawing(node_size=40,
                         edge_width=1,
                         draw_type='bfs'),
            GraphDrawing(node_size=40, edge_width=1,
                         weight_edge_width=2.5,
                         weight_arrow_size=10,
                         draw_type='bfs',
                         colormap=CustomColorMap().cmap)
        )
        # use plasma colormap for the bias
        graph_bias_drawing = (
            GraphDrawing(node_size=40, edge_width=1, colormap=plt.cm.plasma,draw_type='bfs'),
            GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10,draw_type='bfs'),
        )

        texts = []
        labels = []
        tokens = []
        for idx, graph_id in enumerate(graph_ids):
            texts.append(plain_texts[graph_id])
            labels.append(output_labels[graph_id])
            tokens.append(tokenized_texts[graph_id])
            # get convolution layer
            convolution_layer = net.net_layers[0]
            aggregation_layer = net.net_layers[-1]
            convolution_layer.draw(ax=axs[idx][0], graph_id=graph_id, graph_drawing=graph_drawing, graph_only=True)
            convolution_layer.draw(ax=axs[idx][1], graph_id=graph_id, graph_drawing=graph_bias_drawing, graph_only=True, draw_bias_labels=True)
            convolution_layer.draw(ax=axs[idx][2], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None, head=0)
            convolution_layer.draw(ax=axs[idx][3], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights=None, head=1)
            convolution_layer.draw(ax=axs[idx][4], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3}, head=0)
            convolution_layer.draw(ax=axs[idx][5], graph_id=graph_id, graph_drawing=graph_drawing, filter_weights={'absolute': 3}, head=1)

        # add subplots column and row titles
        axs[0][0].set_title(f'Sentence with Word + Position Labels')
        axs[0][1].set_title(f'Sentence with Position labels')
        axs[0][2].set_title(f'All Coefficients (head 0)')
        axs[0][3].set_title(f'All Coefficients (head 1)')
        axs[0][4].set_title(f'Top $3$ Coefficients (head 0)')
        axs[0][5].set_title(f'Top $3$ Coefficients (head 1)')

        for idx, graph_id in enumerate(graph_ids):
            axs[idx][0].set_ylabel(f'Sentence: {graph_id}, Label {net.graph_data.y[graph_id]}, {"Correct" if correct_outputs[idx] else "Wrong"}')

        plt.savefig(f'Examples/TextClassification/Plots/{db_name}.png', bbox_inches='tight', dpi=300)
        plt.show()
        # print the sentences
        for i, (text, label, token) in enumerate(zip(texts, labels, tokens)):
            print(f'Sentence {graph_ids[i]}: {text}')
            print(f'Tokens: {token}')
            print(f'Label: {label} ({"True" if correct_outputs[i] else "False"})')
            print('\n')

    return


if __name__ == '__main__':
    main()