'''
Created on 15.03.2019

@author:
'''
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from src.Preprocessing.create_labels import save_node_labels
from src.Architectures.RuleGNN.RuleGNNLayers import Layer
from src.utils import ReadWriteGraphs as gdtgl
from src.Methods.ModelEvaluation import ModelEvaluation
from src.utils.GraphData import get_graph_data, GraphData
from src.utils.GraphLabels import combine_node_labels, Properties
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_labels import load_labels
from src.utils.load_splits import Load_Splits


def load_preprocessed_data_and_parameters(run_id, validation_id, config_id, validation_folds, graph_db_name, graph_data:GraphData, run_config, para: Parameters):
    experiment_configuration = run_config.config
    # path do db and db
    draw = False
    print_results = False
    save_weights = False
    save_prediction_values = False
    plot_graphs = False
    print_layer_init = False
    # if debug mode is on, turn on all print and draw options
    if experiment_configuration['mode'] == "debug":
        draw = experiment_configuration['additional_options']['draw']
        print_results = experiment_configuration['additional_options']['print_results']
        save_prediction_values = experiment_configuration['additional_options']['save_prediction_values']
        save_weights = experiment_configuration['additional_options']['save_weights']
        plot_graphs = experiment_configuration['additional_options']['plot_graphs']

    for i, l in enumerate(run_config.layers):
        # add the labels to the graph data
        label_path = experiment_configuration['paths']['labels'].joinpath(f"{graph_db_name}_{l.get_layer_string()}_labels.txt")
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[l.get_layer_string()] = g_labels
        elif l.layer_type == "combined":  # create combined file if it is a combined layer and the file does not exist
            combined_labels = []
            # get the labels for each layer in the combined layer
            for x in l.layer_dict['sub_labels']:
                sub_layer = Layer(x, i)
                sub_label_path = experiment_configuration['paths']['labels'].joinpath(f"/{graph_db_name}_{sub_layer.get_layer_string()}_labels.txt")
                if os.path.exists(sub_label_path):
                    g_labels = load_labels(path=sub_label_path)
                    combined_labels.append(g_labels)
                else:
                    # raise an error if the file does not exist
                    raise FileNotFoundError(f"File {sub_label_path} does not exist")
            # combine the labels and save them
            g_labels = combine_node_labels(combined_labels)
            graph_data.node_labels[l.get_layer_string()] = g_labels
            save_node_labels(data_path=experiment_configuration['paths']['labels'], db_names=[graph_db_name], labels=g_labels.node_labels,
                             name=l.get_layer_string(), max_label_num=l.node_labels)
        else:
            # raise an error if the file does not exist and add the absolute path to the error message
            raise FileNotFoundError(f"File {label_path} does not exist")
        # add the properties to the graph data
        if 'properties' in l.layer_dict:
            prop_dict = l.layer_dict['properties']
            prop_name = prop_dict['name']
            if prop_name not in graph_data.properties:
                graph_data.properties[prop_name] = Properties(path=experiment_configuration['paths']['properties'], db_name=graph_db_name,
                                                              property_name=prop_dict['name'],
                                                              valid_values=prop_dict['values'], layer_id=l.layer_id)
            else:
                graph_data.properties[prop_name].add_properties(prop_dict['values'], l.layer_id)
        pass

    """
        BenchmarkGraphs parameters
    """
    para.set_data_param(db=graph_db_name,
                        max_coding=1,
                        layers=run_config.layers, node_features=1,
                        run_config=run_config)

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_folds,
                              validation_id=validation_id,
                              config_id=config_id,
                              n_epochs=run_config.epochs,
                              learning_rate=run_config.lr, dropout=run_config.dropout,
                              balance_data=run_config.config['balance_training'],
                              convolution_grad=True,
                              resize_graph=True)

    """
    Print, save and draw parameters
    """
    para.set_print_param(no_print=False, print_results=print_results, net_print_weights=True, print_number=1,
                         draw=draw, save_weights=save_weights,
                         save_prediction_values=save_prediction_values, plot_graphs=plot_graphs,
                         print_layer_init=print_layer_init)

    """
        Get the first index in the results directory that is not used
    """
    para.set_file_index(size=6)

    if para.plot_graphs:
        # if not exists create the directory
        if not os.path.exists(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots")):
            os.makedirs(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots"))
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots/graph_{str(i).zfill(5)}.png"))


def config_paths_to_absolute(experiment_configuration, absolute_path):
    experiment_configuration['paths']['data'] = absolute_path.joinpath(experiment_configuration['paths']['data'])
    experiment_configuration['paths']['results'] = absolute_path.joinpath(experiment_configuration['paths']['results'])
    experiment_configuration['paths']['labels'] = absolute_path.joinpath(experiment_configuration['paths']['labels'])
    experiment_configuration['paths']['properties'] = absolute_path.joinpath(
        experiment_configuration['paths']['properties'])
    experiment_configuration['paths']['splits'] = absolute_path.joinpath(experiment_configuration['paths']['splits'])


def find_best_models(graph_db_name, validation_id, validation_folds, graph_format, transfer, config, run_id=0):
    if config is not None:
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        absolute_path = Path(absolute_path)
        # read the config yml file
        experiment_configuration = yaml.safe_load(open(absolute_path.joinpath(config)))
        # get the data path from the config file
        # add absolute path to the config data paths
        config_paths_to_absolute(experiment_configuration, absolute_path)
        experiment_configuration['format'] = graph_format
        experiment_configuration['transfer'] = transfer

        data_path = Path(experiment_configuration['paths']['data'])
        r_path = experiment_configuration['paths']['results']

        # if not exists create the results directory
        if not os.path.exists(r_path):
            try:
                os.makedirs(r_path)
            except:
                pass
        # if not exists create the directory for db under Results
        if not os.path.exists(r_path.joinpath(graph_db_name)):
            try:
                os.makedirs(r_path.joinpath(graph_db_name))
            except:
                pass
        # if not exists create the directory Results, Plots, Weights and Models under db
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Results")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Results"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Plots")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Plots"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Weights")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Weights"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Models")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Models"))
            except:
                pass

        plt.ion()

        """
        Create Input data, information and labels from the graphs for training and testing
        """
        graph_data = get_graph_data(db_name=graph_db_name, data_path=data_path, use_features=experiment_configuration['use_features'],
                                    use_attributes=experiment_configuration['use_attributes'], data_format=experiment_configuration['format'])
        # adapt the precision of the input data
        if 'precision' in experiment_configuration:
            if experiment_configuration['precision'] == 'double':
                for i in range(len(graph_data.inputs)):
                    graph_data.inputs[i] = graph_data.inputs[i].double()

        run_configs = get_run_configs(experiment_configuration)

        for config_id, run_config in enumerate(run_configs):
            if 'config_id' in experiment_configuration:
                # if config_id is provided in the config file, add it to the current config_id
                config_id += experiment_configuration['config_id']
            # config_id to string with leading zeros
            c_id = f'Configuration_{str(config_id).zfill(6)}'
            run_configuration(config_id=c_id, run_config=run_config, graph_data=graph_data, graph_db_name=graph_db_name, run_id=run_id, validation_id=validation_id, validation_folds=validation_folds)
        # copy config file to the results directory if it is not already there
        if not os.path.exists(r_path.joinpath(f"{graph_db_name}/config.yml")):
            source_path = Path(absolute_path).joinpath(config)
            destination_path = r_path.joinpath(f"{graph_db_name}/config.yml")
            # copy the config file to the results directory
            # if linux
            if os.name == 'posix':
                os.system(f"cp {source_path} {destination_path}")
            # if windows
            elif os.name == 'nt':
                os.system(f"copy {source_path} {destination_path}")
    else:
        #print that config file is not provided
        print("Please provide a configuration file")


def validation_step(run_id, validation_id, graph_data: GraphData, para: Parameters):
    """
    Split the data in training validation and test set
    """
    seed = 56874687 + validation_id + para.n_val_runs * run_id
    data = Load_Splits(para.splits_path, para.db, para.run_config.config['transfer'])
    test_data = np.asarray(data[0][validation_id], dtype=int)
    training_data = np.asarray(data[1][validation_id], dtype=int)
    validate_data = np.asarray(data[2][validation_id], dtype=int)

    method = ModelEvaluation(run_id, validation_id, graph_data, training_data, validate_data, test_data, seed, para)

    """
    Run the method
    """
    method.Run()


def run_configuration(config_id, run_config, graph_data: GraphData, graph_db_name, run_id, validation_id, validation_folds):
    para = Parameters()
    load_preprocessed_data_and_parameters(config_id=config_id, run_id=run_id, validation_id=validation_id, validation_folds=validation_folds, graph_db_name=graph_db_name, graph_data=graph_data, run_config=run_config, para=para)
    validation_step(para.run_id, para.validation_id, graph_data, para)
