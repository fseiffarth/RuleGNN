import os

from GraphData.GraphData import get_graph_data
from GraphData.Labels.generator.load_labels import load_labels
from Layers.GraphLayers import Layer
from utils.utils import save_graphs


def create_dataset(dataset_name):
    # load the graphs
    data_path = '../../../GraphData/DS_all/'
    distance_path = '../Distances/'
    graph_data = get_graph_data(dataset_name, data_path, distance_path, use_features=True, use_attributes=False)
    # if not exists create the results directory named after the dataset
    if not os.path.exists(f"{dataset_name}"):
        try:
            os.makedirs(f"{dataset_name}")
        except:
            pass
    output_path = f"{dataset_name}/"

    layers = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500}), Layer({'layer_type': 'simple_cycles', 'max_cycle_length': 10})]
    for l in layers:
        label_path = f"../Labels/{dataset_name}_{l.get_layer_string()}_labels.txt"
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            # add g_labels as node attributes to the graph_data
            for i, g in enumerate(graph_data.graphs):
                node_labels = g_labels.node_labels[i]
                for node in g.nodes:
                    if 'attr' in g.nodes[node]:
                        g.nodes[node]['attr'].append(node_labels[node])
                    else:
                        g.nodes[node]['attr'] = [node_labels[node]]
                    # delete the attribute key and value from the node data dict
                    g.nodes[node].pop('attribute', None)
    # graph_labels to 0,1,2 ...
    # if there exist graph labels -1, 1 shift to 0,1
    if min(graph_data.graph_labels) == -1 and max(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] += 1
            graph_data.graph_labels[i] //= 2


    save_graphs(path=output_path, db_name=dataset_name, graphs=graph_data.graphs, labels=graph_data.graph_labels)


def main():
    create_dataset('DHFR')


if __name__ == "__main__":
    main()
