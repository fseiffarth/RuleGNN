import os

from utils.GraphData import get_graph_data
from Preprocessing.load_labels import load_labels
from Architectures.RuleGNN.RuleGNNLayers import Layer
from utils.utils import save_graphs


def create_dataset(dataset_name, layers=None, with_degree=False):
    # load the graphs
    data_path = '../../GraphData/DS_all/'
    distance_path = '../Data/Distances/'
    graph_data = get_graph_data(dataset_name, data_path, distance_path, use_features=True, use_attributes=False)
    output_path = ""


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
    # if the graph labels start from 1, shift to 0,1,2 ...
    if min(graph_data.graph_labels) == 1:
        for i, label in enumerate(graph_data.graph_labels):
            graph_data.graph_labels[i] -= 1


    save_graphs(path=output_path, db_name=f'{dataset_name}Features', graphs=graph_data.graphs, labels=graph_data.graph_labels, with_degree=with_degree)
    # copy the split data in the processed folder and rename it to dataset_nameFeatures_splits.json
    os.system(f"cp ../DataSplits/{dataset_name}_splits.json {output_path}{dataset_name}Features/processed/{dataset_name}Features_splits.json")


def main():
    imdb_multi_layers  = [Layer({'layer_type': 'subgraph', id:1}), Layer({'layer_type': 'subgraph', id:1})]
    create_dataset('IMDB-MULTI', layers=imdb_multi_layers, with_degree=True)
    imdb_binary_layers  = [Layer({'layer_type': 'subgraph', id:1}), Layer({'layer_type': 'induced_cycles', 'max_cycle_length': 5})]
    create_dataset('IMDB-BINARY', layers=imdb_binary_layers, with_degree=True)

    mutagenicity_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500}), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000})]
    create_dataset('Mutagenicity', layers=mutagenicity_layers)
    dhfr_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500}), Layer({'layer_type': 'simple_cycles', 'max_cycle_length': 10})]
    create_dataset('DHFR', layers=dhfr_layers)
    nci1_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500}), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000})]
    create_dataset('NCI1', layers=nci1_layers)
    nci109_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500}), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000})]
    create_dataset('NCI109', layers=nci109_layers)




if __name__ == "__main__":
    main()
