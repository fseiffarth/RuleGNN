from GraphData.GraphData import get_graph_data
from Layers.GraphLayers import Layer


def create_dataset(dataset_name):
    # load the graphs
    data_path = '../../../GraphData/DS_all/'
    distance_path = '../Distances/'
    graph_data = get_graph_data(dataset_name, data_path, distance_path, use_features=True,use_attributes=False)

    layers = [Layer({'layer_type' : 'wl', 'wl_iterations' : 2, 'max_node_labels': 500})]

    label_path = f"GraphData/Labels/{dataset_name}_{l.get_layer_string()}_labels.txt"
    for label_path in label_paths:
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[l.get_layer_string()] = g_labels


def main():
    create_dataset('NCI1')

if __name__ == "__main__":
    main()