import os
from pathlib import Path
from src.utils.GraphData import get_graph_data
from src.utils.load_labels import load_labels
from src.Architectures.RuleGNN.RuleGNNLayers import Layer
from src.utils.utils import save_graphs


def create_dataset(dataset_name, paths:dict[str, Path], output_path:Path, layers=None, with_degree=False):
    # load the graphs
    data_path = paths['data']
    label_path = paths['labels']
    splits_path = paths['splits']
    graph_data = get_graph_data(dataset_name, data_path, use_labels=True, use_attributes=False, graph_format='NEL')


    for l in layers:
        layer_label_path = label_path.joinpath(f"{dataset_name}_{l.get_layer_string()}_labels.txt")
        if os.path.exists(layer_label_path):
            g_labels = load_labels(path=layer_label_path)
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


    save_graphs(path=output_path, db_name=f'{dataset_name}Features', graphs=graph_data.graphs, labels=graph_data.graph_labels, with_degree=with_degree, graph_format='NEL')
    # copy the split data in the processed folder and rename it to dataset_nameFeatures_splits.json
    source_path = splits_path.joinpath(f"{dataset_name}_splits.json")
    target_path = output_path.joinpath(f"{dataset_name}Features/processed/{dataset_name}Features_splits.json")
    os.system(f"cp {source_path} {target_path}")


def main():
    synthetic_paths = {'data': Path(f'Reproduce_RuleGNN/Data/SyntheticDatasets/'),
                'labels': Path(f'Reproduce_RuleGNN/Data/Labels/'),
                'splits': Path(f'Reproduce_RuleGNN/Data/Splits/')}
    tu_paths = {'data': Path(f'Reproduce_RuleGNN/Data/TUDatasets/'),
                'labels': Path(f'Reproduce_RuleGNN/Data/Labels/'),
                'splits': Path(f'Reproduce_RuleGNN/Data/Splits/')}

    snowflakes_layers = [Layer({'layer_type': 'subgraph', 'id': 0}, 0), Layer({'layer_type': 'wl', 'wl_iterations': 0}, 1)]
    create_dataset('Snowflakes', paths=synthetic_paths, output_path=Path(f'Data/Featured/'), layers=snowflakes_layers)

    csl_layers = [Layer({'layer_type': 'simple_cycles', 'max_cycle_length': 10}, 0)]
    create_dataset('CSL', paths=synthetic_paths, output_path=Path(f'Data/Featured/'), layers=csl_layers)

    imdb_multi_layers  = [Layer({'layer_type': 'subgraph', 'id':1},0)]
    create_dataset('IMDB-MULTI', paths=tu_paths, output_path=Path(f'Data/Featured/'), layers=imdb_multi_layers)
    imdb_binary_layers  = [Layer({'layer_type': 'subgraph', 'id':1},0), Layer({'layer_type': 'induced_cycles', 'max_cycle_length': 5},1)]
    create_dataset('IMDB-BINARY', paths=tu_paths, output_path=Path(f'Data/Featured/'), layers=imdb_binary_layers)

    mutagenicity_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500},0), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000},1)]
    create_dataset('Mutagenicity', paths=tu_paths, output_path=Path(f'Data/Featured/'), layers=mutagenicity_layers)

    nci1_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500},0), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000},0)]
    create_dataset('NCI1', layers=nci1_layers, paths=tu_paths, output_path=Path(f'Data/Featured/'))

    nci109_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500},0), Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 50000},0)]
    create_dataset('NCI109', layers=nci109_layers, paths=tu_paths, output_path=Path(f'Data/Featured/'))

    dhfr_layers  = [Layer({'layer_type': 'wl', 'wl_iterations': 2, 'max_node_labels': 500},0), Layer({'layer_type': 'simple_cycles', 'max_cycle_length': 10},1)]
    create_dataset('DHFR', layers=dhfr_layers , paths=tu_paths, output_path=Path(f'Data/Featured/'))




if __name__ == "__main__":
    main()
