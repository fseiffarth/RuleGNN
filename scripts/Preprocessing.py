import json
from pathlib import Path

import yaml

from src.Preprocessing.create_labels import save_trivial_labels, save_wl_labels, save_primary_labels, \
    save_degree_labels, save_cycle_labels
from src.Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from src.Preprocessing.create_splits import create_splits
from src.utils.RunConfiguration import get_run_configs
from src.utils.utils import save_graphs


class Preprocessing:
    def __init__(self, db_name:str, config_file: Path, with_splits=True, get_data=None, get_data_args=None):
        self.db_name = db_name
        # load the config file
        self.configs = yaml.safe_load(open(config_file))
        # create config folders if they do not exist
        Path(self.configs['paths']['data']).mkdir(exist_ok=True)
        Path(self.configs['paths']['labels']).mkdir(exist_ok=True)
        Path(self.configs['paths']['properties']).mkdir(exist_ok=True)
        Path(self.configs['paths']['splits']).mkdir(exist_ok=True)
        Path(self.configs['paths']['results']).mkdir(exist_ok=True)

        if type(get_data) == str:
            # download data
            pass
        else:
            # generate data
            graphs, labels = get_data(*get_data_args)
            # save lists of graphs and labels in the correct format NEL -> Nodes, Edges, Labels
            save_graphs(Path(self.configs['paths']['data']), self.db_name, graphs, labels, with_degree=False, format='NEL')


        if with_splits:
            # create the splits folder if it does not exist
            Path(self.configs['paths']['splits']).mkdir(exist_ok=True)
            # generate splits
            create_splits(db_name, Path(self.configs['paths']['data']), Path(self.configs['paths']['splits']), folds=10, format='NEL')

        self.preprocessing_from_config()



    def layer_to_labels(self, layer_strings: json):
        layer = json.loads(layer_strings)
        # switch case for the different layers
        if layer['layer_type'] == 'primary':
            save_primary_labels(Path(self.configs['paths']['data']), db_names=[self.db_name], label_path=Path(self.configs['paths']['labels']), format='NEL')
        elif layer['layer_type'] == 'trivial':
            save_trivial_labels(Path(self.configs['paths']['data']), db_names=[self.db_name], label_path=Path(self.configs['paths']['labels']), format='NEL')
        elif layer['layer_type'] == 'wl':
            if 'max_node_labels' not in layer:
                layer['max_node_labels'] = None
            if 'wl_iterations' not in layer:
                layer['wl_iterations'] = None
            if 'wl_iterations' in layer and layer['wl_iterations'] == 0:
                save_degree_labels(Path(self.configs['paths']['data']), db_names=[self.db_name], label_path=Path(self.configs['paths']['labels']), format='NEL')
            else:
                save_wl_labels(Path(self.configs['paths']['data']), db_names=[self.db_name], max_iterations=layer['wl_iterations'], max_label_num=layer['max_node_labels'], label_path=Path(self.configs['paths']['labels']), format='NEL')
        elif layer['layer_type'] == 'simple_cycles' or layer['layer_type'] == 'induced_cycles':
            cycle_type = 'simple' if layer['layer_type'] == 'simple_cycles' else 'induced'
            if 'max_node_labels' not in layer:
                layer['max_node_labels'] = None
            if 'max_cycle_length' not in layer:
                layer['max_cycle_length'] = None
            save_cycle_labels(Path(self.configs['paths']['data']), db_names=[self.db_name], length_bound=layer['max_cycle_length'], cycle_type=cycle_type, label_path=Path(self.configs['paths']['labels']), format='NEL')
        else:
            pass
        pass
    def property_to_properties(self, property_strings: json):
        # switch case for the different properties
        properties = json.loads(property_strings)
        if properties['name'] == 'distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_properties(Path(self.configs['paths']['data']), db_name=self.db_name, out_path=Path(self.configs['paths']['properties']), cutoff=properties['cutoff'], data_format='NEL')
        elif properties['name'] == 'edge_label_distances':
            if 'cutoff' not in properties:
                properties['cutoff'] = None
            write_distance_edge_properties(Path(self.configs['paths']['data']), db_name=self.db_name, out_path=Path(self.configs['paths']['properties']), cutoff=properties['cutoff'], data_format='NEL')

    # generate preprocessing by scanning the config file
    def preprocessing_from_config(self):
        # get the layers from the config file
        run_configs = get_run_configs(self.configs)
        # preprocessed layers
        preprocessed_layers = set()
        preprocessed_properties = set()
        # iterate over the layers
        for run_config in run_configs:
            for layer in run_config.layers:
                if 'properties' in layer.layer_dict:
                    properties = layer.layer_dict.pop('properties')
                    json_properties = json.dumps(properties, sort_keys=True)
                    preprocessed_properties.add(json_properties)
                json_layer = json.dumps(layer.layer_dict, sort_keys=True)
                preprocessed_layers.add(json_layer)
        # generate all necessary labels and properties
        for layer in preprocessed_layers:
            self.layer_to_labels(layer)
        for property in preprocessed_properties:
            self.property_to_properties(property)



