import json
import os
from pathlib import Path

import torch
from torch.cuda import graph
from torch_geometric.datasets import TUDataset
from src.Preprocessing.create_labels import save_trivial_labels, save_wl_labels, save_primary_labels, \
    save_degree_labels, save_cycle_labels, save_subgraph_labels, save_clique_labels, save_index_labels, \
    save_labeled_degree_labels, save_wl_labeled_labels, save_labels_to_file
from src.Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from src.Preprocessing.create_splits import create_splits
from src.utils.GraphData import get_graph_data, RuleGNNDataset
from src.utils.GraphLabels import combine_node_labels
from src.utils.RunConfiguration import get_run_configs
from src.utils.TU_to_NEL import tu_to_nel
from src.utils.load_labels import load_labels
from src.utils.utils import save_graphs
import networkx as nx


class Preprocessing:
    def __init__(self, db_name:str, dataset_configuration, experiment_configuration, with_splits=True, with_labels_and_properties=True, data_generation=None, data_generation_args=None, create_pt_files = True):
        self.db_name = db_name
        self.graph_data = None
        # load the config file
        self.experiment_configuration = experiment_configuration
        self.dataset_configuration = dataset_configuration
        # create config folders if they do not exist
        self.experiment_configuration['paths']['data'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['labels'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['properties'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['splits'].mkdir(exist_ok=True, parents=True)
        self.experiment_configuration['paths']['results'].mkdir(exist_ok=True, parents=True)

        # if not exists create the generation_times_labels.txt and generation_times_properties.txt in the Results folder
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt'), 'w') as f:
                f.write('Generation times for labels\n')
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt'), 'w') as f:
                f.write('Generation times for properties\n')
        self.generation_times_labels_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_labels.txt')
        self.generation_times_properties_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_properties.txt')

        # generate the data only if it does not exist (i.e. the processed folder is empty)
        nel_dataset = False
        if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{db_name}').joinpath('processed').joinpath(f'data.pt').is_file():
            if isinstance(data_generation, str):
                if data_generation != 'generate_from_function':
                    try:
                        path = Path(self.experiment_configuration['paths']['data'])
                        if Path(Path(self.experiment_configuration['paths']['data']) / db_name / 'raw').exists() and len(
                                list(Path(Path(self.experiment_configuration['paths']['data']) / db_name / 'raw').iterdir())) > 0:
                            print(f"Dataset {db_name} already exists in {Path(self.experiment_configuration['paths']['data'])} . Skip the data generation.")
                            return
                        # download the dataset
                        # create a tmp folder to store the dataset
                        if not Path('tmp').exists():
                            Path('tmp').mkdir()
                        self.graph_data = RuleGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                                         name=db_name,
                                                         from_existing_data=data_generation,
                                                         )
                        if not os.path.exists(path.joinpath(Path(db_name))):
                            os.makedirs(path.joinpath(Path(db_name)))
                        # create processed and raw folders in path+db_name
                        if not os.path.exists(path.joinpath(Path(db_name + "/processed"))):
                            os.makedirs(path.joinpath(Path(db_name + "/processed")))
                        if not os.path.exists(path.joinpath(Path(db_name + "/raw"))):
                            os.makedirs(path.joinpath(Path(db_name + "/raw")))
                        #tu_to_nel(db_name=db_name, out_path=Path(self.experiment_configuration['paths']['data']))
                    except:
                        print(f'Could not generate {db_name} from TUDataset')
                else:
                    print(f'Do not know how to handle data from {data_generation}. Do you mean "TUDataset"?')
                pass
            else:
                # TODO generate the pt data
                if data_generation is not None:
                    if data_generation_args is None:
                        data_generation_args = {}
                    try:
                        # generate data
                        graphs, labels =  data_generation(**data_generation_args, split_path=Path(self.experiment_configuration['paths']['splits']))
                        # save lists of graphs and labels in the correct graph_format NEL -> Nodes, Edges, Labels
                        save_graphs(Path(self.experiment_configuration['paths']['data']), self.db_name, graphs, labels, with_degree=False, graph_format='NEL')
                        self.graph_data = RuleGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                                         name=db_name,
                                                         use_node_attr=self.experiment_configuration.get(
                                                             'use_node_attr', False),
                                                         use_edge_attr=self.experiment_configuration.get(
                                                             'use_edge_attr', False),
                                                         delete_zero_columns=self.experiment_configuration.get(
                                                             'delete_zero_columns', True),
                                                        from_existing_data='NEL'
                                                         )
                    except:
                        # raise the error that has occurred
                        print(f'Could not generate {db_name} from function {data_generation} with arguments {data_generation_args}')


        # load the graph data TODO: introduce new pyg format and load from the pt files
        #self.graph_data = get_graph_data(db_name=self.db_name,
        #                                 data_path=self.experiment_configuration['paths']['data'],
        #                                 graph_format='NEL',
        #                                 only_graphs=True)

        # load graph data from pt files if it exists in the processed folder
        if self.graph_data is None and self.experiment_configuration['paths']['data'].joinpath(f'{db_name}').joinpath('processed').exists():
            self.graph_data = RuleGNNDataset(root=str(self.experiment_configuration['paths']['data']),
                                             name=db_name,
                                             use_node_attr=self.experiment_configuration.get('use_node_attr', False),
                                                use_edge_attr=self.experiment_configuration.get('use_edge_attr', False),
                                             delete_zero_columns=self.experiment_configuration.get('delete_zero_columns', True),
                                             )

        # generate the splits
        if with_splits:
            # create the splits folder if it does not exist
            Path(self.experiment_configuration['paths']['splits']).mkdir(exist_ok=True)
            # generate splits
            create_splits(db_name, Path(self.experiment_configuration['paths']['data']), Path(self.experiment_configuration['paths']['splits']), folds=self.dataset_configuration['validation_folds'], graph_data=self.graph_data)

        # copy the splits to the processed folder
        if self.experiment_configuration['paths']['splits'].joinpath(f'{db_name}_splits.json').exists():
            split_file_path = self.experiment_configuration['paths']['splits'].joinpath(f'{db_name}_splits.json')
            if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{db_name}').joinpath('processed').exists():
                Path(self.experiment_configuration['paths']['data']).joinpath(f'{db_name}').joinpath('processed').mkdir()
            split_target_path = Path(self.experiment_configuration['paths']['data']).joinpath(f'{db_name}').joinpath('processed').joinpath(f'{db_name}_splits.json')
            # copy the content of the split file to the target path
            split_target_path.write_text(split_file_path.read_text())

        # generate the labels and properties automatically from the config file
        if with_labels_and_properties:
            self.preprocessing_from_config()



    def layer_to_labels(self, layer_strings: json)->Path:
        file_path = None
        layer = json.loads(layer_strings)
        label_path = self.experiment_configuration['paths']['labels'].joinpath(f'{self.graph_data.name}')
        # check if the path exists, otherwise create it
        if not label_path.exists():
            label_path.mkdir()
        # if label_type is a list, then the layer is a combination of different label types
        if type(layer['label_type']) == list and len(layer['label_type']) > 1:
            # recursively call the function for each label type
            labels = []
            label_names = []
            for label_type in layer['label_type']:
                new_layer_string = layer.copy()
                # remove the label_type key and replace it with the new label_type
                new_layer_string['label_type'] = label_type
                l_path = self.layer_to_labels(json.dumps(new_layer_string))
                # get all after last /
                label_name = '_'.join(l_path.stem.split('_')[1:-1])
                label_names.append(label_name)
                labels.append(load_labels(l_path))
            # combine the labels
            combined_labels = combine_node_labels(labels)
            l = f'{combined_labels.label_name}'
            max_labels = layer.get('max_labels', None)
            if max_labels is not None:
                l += f'_{max_labels}'
            file_path = label_path.joinpath(f"{self.graph_data.name}_labels_{l}.pt")
            save_labels_to_file(file_path, combined_labels.dataset_name, l, combined_labels.node_labels, max_labels=layer.get('max_labels', None))
        else:
            if isinstance(layer['label_type'], list):
                layer['label_type'] = layer['label_type'][0]
            # switch case for the different layers
            if layer['label_type'] == 'primary':
                file_path = save_primary_labels(graph_data=self.graph_data,
                                                label_path=label_path,
                                                max_labels=layer.get('max_labels', None),
                                                save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'trivial':
                file_path = save_trivial_labels(graph_data=self.graph_data,
                                                label_path=label_path,
                                                save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'index':
                file_path = save_index_labels(graph_data=self.graph_data,
                                              max_labels=layer.get('max_labels', None),
                                              label_path=label_path,
                                              save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'degree':
                file_path = save_degree_labels(graph_data=self.graph_data,
                                               label_path=label_path,
                                               max_labels=layer.get('max_labels', None),
                                               save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'wl':
                layer['max_labels'] = layer.get('max_labels', None)
                layer['depth'] = layer.get('depth', 3)
                if layer['depth'] == 0:
                    file_path = save_degree_labels(graph_data=self.graph_data,
                                                   label_path=label_path,
                                                   max_labels=layer.get('max_labels', None),
                                                   save_times=self.generation_times_labels_path)
                else:
                    file_path = save_wl_labels(graph_data=self.graph_data,
                                               depth=layer.get('depth', 3),
                                               max_labels=layer['max_labels'],
                                               label_path=label_path,
                                               save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'wl_labeled':
                layer['max_labels'] = layer.get('max_labels', None)
                layer['depth'] = layer.get('depth', 3)
                if layer['depth'] == 0:
                    file_path = save_labeled_degree_labels(graph_data=self.graph_data,
                                                           label_path=label_path,
                                                              max_labels=layer.get('max_labels', None),
                                                           save_times=self.generation_times_labels_path)
                else:
                    file_path = save_wl_labeled_labels(graph_data=self.graph_data,
                                                       depth=layer.get('depth', 3),
                                                       max_labels=layer['max_labels'],
                                                       label_path=label_path,
                                                       save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'simple_cycles' or layer['label_type'] == 'induced_cycles':
                cycle_type = 'simple' if layer['label_type'] == 'simple_cycles' else 'induced'
                if 'max_labels' not in layer:
                    layer['max_labels'] = None
                if 'max_cycle_length' not in layer:
                    layer['max_cycle_length'] = None
                file_path = save_cycle_labels(graph_data=self.graph_data,
                                              length_bound=layer['max_cycle_length'],
                                              max_labels=layer["max_labels"],
                                              cycle_type=cycle_type,
                                              label_path=label_path,
                                              save_times=self.generation_times_labels_path)
            elif layer['label_type'] == 'subgraph':
                if 'id' in layer:
                    if layer['id'] > len(self.experiment_configuration['subgraphs']):
                        raise ValueError(f'Please specify the subgraphs in the config files under the key "subgraphs" as folllows: subgraphs: - "[nx.complete_graph(4)]"')
                    else:
                        subgraph_list = eval(self.experiment_configuration['subgraphs'][layer['id']])
                        file_path = save_subgraph_labels(graph_data=self.graph_data,
                                                         subgraphs=subgraph_list,
                                                         subgraph_id=layer['id'],
                                                         max_labels=layer.get('max_labels', None),
                                                         label_path=label_path,
                                                         save_times=self.generation_times_labels_path)
                else:
                    raise ValueError(f'Please specify the id of the subgraph in the layer with description {layer_strings}.')
            elif layer['label_type'] == 'cliques':
                if 'max_labels' not in layer:
                    layer['max_labels'] = None
                if 'max_clique_size' not in layer:
                    layer['max_clique_size'] = None
                file_path = save_clique_labels(graph_data=self.graph_data,
                                               max_clique=layer['max_clique_size'],
                                               max_labels=layer.get('max_labels', None),
                                               label_path=label_path,
                                               save_times=self.generation_times_labels_path)
            else:
                # print in red in the console
                print(f'The automatic generation of labels for the layer type {layer["label_type"]} is not supported yet.')
        return file_path

    def property_to_properties(self, property_strings: json):
        properties_path = self.experiment_configuration['paths']['properties'].joinpath(f'{self.graph_data.name}')
        # check if the path exists, otherwise create it
        if not properties_path.exists():
            properties_path.mkdir()
        # switch case for the different properties
        properties = json.loads(property_strings)
        if properties['name'] == 'distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_properties(self.graph_data, out_path=properties_path, cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)
        # TODO: change the edge_label_distances to the new torch format
        elif properties['name'] == 'edge_label_distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_edge_properties(self.graph_data, out_path=properties_path, cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)

    # generate preprocessing by scanning the config file
    def preprocessing_from_config(self):
        # get the layers from the config file
        run_configs = get_run_configs(self.experiment_configuration)
        # preprocessed layers
        preprocessed_label_dicts = set()
        preprocessed_properties = set()
        # iterate over the layers
        for run_config in run_configs:
            for layer in run_config.layers:
                for property_dict in layer.get_unique_property_dicts():
                    p_dict = property_dict.copy()
                    p_dict.pop('values')
                    json_property = json.dumps(p_dict, sort_keys=True)
                    preprocessed_properties.add(json_property)
                for label_dict in layer.get_unique_layer_dicts():
                    json_layer = json.dumps(label_dict, sort_keys=True)
                    preprocessed_label_dicts.add(json_layer)
        # generate all necessary labels and properties, first need to create the nx graphs to run the algorithms on
        self.graph_data.create_nx_graphs(directed=False)
        for layer in preprocessed_label_dicts:
            self.layer_to_labels(layer)
        for preprocessed_property in preprocessed_properties:
            self.property_to_properties(preprocessed_property)





