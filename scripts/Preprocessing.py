import json
from pathlib import Path

from torch.cuda import graph

from src.Preprocessing.create_labels import save_trivial_labels, save_wl_labels, save_primary_labels, \
    save_degree_labels, save_cycle_labels, save_subgraph_labels, save_clique_labels, save_index_labels, \
    save_wl_labeled_labels, save_labeled_degree_labels
from src.Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from src.Preprocessing.create_splits import create_splits
from src.utils.GraphData import get_graph_data
from src.utils.RunConfiguration import get_run_configs
from src.utils.TU_to_NEL import tu_to_nel
from src.utils.utils import save_graphs
import networkx as nx


class Preprocessing:
    def __init__(self, db_name:str, dataset_configuration, experiment_configuration, with_splits=True, with_labels_and_properties=True, data_generation=None, data_generation_args=None, create_pt_files = True):
        self.db_name = db_name
        # load the config file
        self.experiment_configuration = experiment_configuration
        self.dataset_configuration = dataset_configuration
        # create config folders if they do not exist
        self.experiment_configuration['paths']['data'].mkdir(exist_ok=True)
        self.experiment_configuration['paths']['labels'].mkdir(exist_ok=True)
        self.experiment_configuration['paths']['properties'].mkdir(exist_ok=True)
        self.experiment_configuration['paths']['splits'].mkdir(exist_ok=True)
        self.experiment_configuration['paths']['results'].mkdir(exist_ok=True)

        # if not exists create the generation_times_labels.txt and generation_times_properties.txt in the Results folder
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_labels.txt'), 'w') as f:
                f.write('Generation times for labels\n')
        if not Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt').exists():
            with open(Path(self.experiment_configuration['paths']['results']).joinpath('generation_times_properties.txt'), 'w') as f:
                f.write('Generation times for properties\n')
        self.generation_times_labels_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_labels.txt')
        self.generation_times_properties_path = self.experiment_configuration['paths']['results'].joinpath('generation_times_properties.txt')

        # generate the data only if it does not exist
        if not Path(self.experiment_configuration['paths']['data']).joinpath(f'{db_name}').joinpath('raw').joinpath(f'{db_name}_Edges.txt').exists():
            if type(data_generation) == str:
                if data_generation == 'TUDataset':
                    try:
                        tu_to_nel(db_name=db_name, out_path=Path(self.experiment_configuration['paths']['data']))
                    except:
                        print(f'Could not generate {db_name} from TUDataset')
                else:
                    print(f'Do not know how to handle data from {data_generation}. Do you mean "TUDataset"?')
                pass
            else:
                if data_generation is not None:
                    if data_generation_args is None:
                        data_generation_args = {}
                    try:
                        # generate data
                        graphs, labels =  data_generation(**data_generation_args, split_path=Path(self.experiment_configuration['paths']['splits']))
                        # save lists of graphs and labels in the correct graph_format NEL -> Nodes, Edges, Labels
                        save_graphs(Path(self.experiment_configuration['paths']['data']), self.db_name, graphs, labels, with_degree=False, graph_format='NEL')
                    except:
                        # raise the error that has occurred
                        print(f'Could not generate {db_name} from function {data_generation} with arguments {data_generation_args}')


        self.graph_data = get_graph_data(db_name=self.db_name, data_path=self.experiment_configuration['paths']['data'], graph_format='NEL')

        if create_pt_files:
            pass # TODO: create pt files

        if with_splits:
            # create the splits folder if it does not exist
            Path(self.experiment_configuration['paths']['splits']).mkdir(exist_ok=True)
            # generate splits
            create_splits(db_name, Path(self.experiment_configuration['paths']['data']), Path(self.experiment_configuration['paths']['splits']), folds=self.dataset_configuration['validation_folds'], graph_format='NEL')

        if with_labels_and_properties:
            # creates the labels and properties automatically
            self.preprocessing_from_config()



    def layer_to_labels(self, layer_strings: json):
        layer = json.loads(layer_strings)
        # switch case for the different layers
        if layer['label_type'] == 'primary':
            save_primary_labels(graph_data=self.graph_data, label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'trivial':
            save_trivial_labels(graph_data=self.graph_data, label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'index':
            save_index_labels(graph_data=self.graph_data, max_labels=layer.get('max_labels', None), label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'degree':
            save_degree_labels(graph_data=self.graph_data, label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'wl':
            layer['max_labels'] = layer.get('max_labels', None)
            layer['depth'] = layer.get('depth', 3)
            if layer['depth'] == 0:
                save_degree_labels(graph_data=self.graph_data, label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
            else:
                save_wl_labels(graph_data=self.graph_data, depth=layer.get('depth', 3), max_labels=layer['max_labels'], label_path=Path(self.experiment_configuration['paths']['labels']),  save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'wl_labeled':
            layer['max_labels'] = layer.get('max_labels', None)
            layer['depth'] = layer.get('depth', 3)
            if layer['depth'] == 0:
                save_labeled_degree_labels(graph_data=self.graph_data, label_path=Path(self.experiment_configuration['paths']['labels']), save_times=self.generation_times_labels_path)
            else:
                save_wl_labeled_labels(graph_data=self.graph_data, depth=layer.get('depth', 3), max_labels=layer['max_labels'], label_path=Path(self.experiment_configuration['paths']['labels']),  save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'simple_cycles' or layer['label_type'] == 'induced_cycles':
            cycle_type = 'simple' if layer['label_type'] == 'simple_cycles' else 'induced'
            if 'max_labels' not in layer:
                layer['max_labels'] = None
            if 'max_cycle_length' not in layer:
                layer['max_cycle_length'] = None
            save_cycle_labels(graph_data=self.graph_data, length_bound=layer['max_cycle_length'], max_labels=layer["max_labels"], cycle_type=cycle_type, label_path=Path(self.experiment_configuration['paths']['labels']),  save_times=self.generation_times_labels_path)
        elif layer['label_type'] == 'subgraph':
            if 'id' in layer:
                if layer['id'] > len(self.experiment_configuration['subgraphs']):
                    raise ValueError(f'Please specify the subgraphs in the config files under the key "subgraphs" as folllows: subgraphs: - "[nx.complete_graph(4)]"')
                else:
                    subgraph_list = eval(self.experiment_configuration['subgraphs'][layer['id']])
                    save_subgraph_labels(graph_data=self.graph_data, subgraphs=subgraph_list, id=layer['id'], label_path=Path(self.experiment_configuration['paths']['labels']),  save_times=self.generation_times_labels_path)
            else:
                raise ValueError(f'Please specify the id of the subgraph in the layer with description {layer_strings}.')
        elif layer['label_type'] == 'cliques':
            if 'max_labels' not in layer:
                layer['max_labels'] = None
            if 'max_clique_size' not in layer:
                layer['max_clique_size'] = None
            save_clique_labels(graph_data=self.graph_data, max_clique=layer['max_clique_size'], max_labels=layer['max_labels'], label_path=Path(self.experiment_configuration['paths']['labels']),  save_times=self.generation_times_labels_path)
        else:
            # print in red in the console
            print(f'The automatic generation of labels for the layer type {layer["label_type"]} is not supported yet.')


    def property_to_properties(self, property_strings: json):
        # switch case for the different properties
        properties = json.loads(property_strings)
        if properties['name'] == 'distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_properties(self.graph_data, out_path=Path(self.experiment_configuration['paths']['properties']), cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)
        elif properties['name'] == 'edge_label_distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_edge_properties(self.graph_data, out_path=Path(self.experiment_configuration['paths']['properties']), cutoff=properties['cutoff'],  save_times=self.generation_times_properties_path)

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
                    property_dict.pop('values')
                    json_property = json.dumps(property_dict, sort_keys=True)
                    preprocessed_properties.add(json_property)
                for label_dict in layer.get_unique_layer_dicts():
                    json_layer = json.dumps(label_dict, sort_keys=True)
                    preprocessed_label_dicts.add(json_layer)
        # generate all necessary labels and properties
        for layer in preprocessed_label_dicts:
            self.layer_to_labels(layer)
        for preprocessed_property in preprocessed_properties:
            self.property_to_properties(preprocessed_property)





