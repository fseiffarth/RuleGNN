'''
Created on 15.03.2019

@author:
'''
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import yaml

from utils.load_splits import Load_Splits
from utils.GraphData import get_graph_data
from utils.load_labels import load_labels
from Preprocessing.create_labels import save_node_labels
from Architectures.RuleGNN.RuleGNNLayers import Layer
from utils import GraphData, ReadWriteGraphs as gdtgl
from Methods.ModelEvaluation import ModelEvaluation
from utils.GraphLabels import combine_node_labels, Properties
from utils.Parameters import Parameters
from utils.RunConfiguration import RunConfiguration


def get_run_configs(configs):
    # define the network type from the config file
    run_configs = []
    task = "classification"
    if 'task' in configs:
        task = configs['task']
    # iterate over all network architectures
    for network_architecture in configs['networks']:
        layers = []
        # get all different run configurations
        for i, l in enumerate(network_architecture):
            layers.append(Layer(l, i))
        for b in configs['batch_size']:
            for lr in configs['learning_rate']:
                for e in configs['epochs']:
                    for d in configs['dropout']:
                        for o in configs['optimizer']:
                            for loss in configs['loss']:
                                run_configs.append(
                                    RunConfiguration(network_architecture, layers, b, lr, e, d, o, loss, task))
    return run_configs


@click.command()
@click.option('--graph_db_name', default="MUTAG", type=str, help='Database name')
@click.option('--validation_number', default=10, type=int)
@click.option('--validation_id', default=0, type=int)
@click.option('--config', default=None, type=str)
# current configuration
#--graph_db_name NCI1 --config Configs/config_NCI1_test.yml --validation_number 10 --validation_id 0

def main(graph_db_name, validation_number, validation_id, config, run_id=0):
    if config is not None:
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        # read the config yml file
        configs = yaml.safe_load(open(absolute_path + "/" + config))
        # get the data path from the config file
        # add absolute path to the config data paths
        configs['paths']['data'] = absolute_path + "/" + configs['paths']['data']
        configs['paths']['results'] = absolute_path + "/" + configs['paths']['results']
        configs['paths']['labels'] = absolute_path + "/" + configs['paths']['labels']
        configs['paths']['properties'] = absolute_path + "/" + configs['paths']['properties']
        configs['paths']['splits'] = absolute_path + "/" + configs['paths']['splits']

        data_path = configs['paths']['data']
        r_path = configs['paths']['results']

        # if not exists create the results directory
        if not os.path.exists(r_path):
            try:
                os.makedirs(r_path)
            except:
                pass
        # if not exists create the directory for db under Results
        if not os.path.exists(r_path + graph_db_name):
            try:
                os.makedirs(r_path + graph_db_name)
            except:
                pass
        # if not exists create the directory Results, Plots, Weights and Models under db
        if not os.path.exists(r_path + graph_db_name + "/Results"):
            try:
                os.makedirs(r_path + graph_db_name + "/Results")
            except:
                pass
        if not os.path.exists(r_path + graph_db_name + "/Plots"):
            try:
                os.makedirs(r_path + graph_db_name + "/Plots")
            except:
                pass
        if not os.path.exists(r_path + graph_db_name + "/Weights"):
            try:
                os.makedirs(r_path + graph_db_name + "/Weights")
            except:
                pass
        if not os.path.exists(r_path + graph_db_name + "/Models"):
            try:
                os.makedirs(r_path + graph_db_name + "/Models")
            except:
                pass

        plt.ion()

        """
        Create Input data, information and labels from the graphs for training and testing
        """
        graph_data = get_graph_data(db_name=graph_db_name, data_path=data_path, use_features=configs['use_features'],
                                    use_attributes=configs['use_attributes'])
        # adapt the precision of the input data
        if 'precision' in configs:
            if configs['precision'] == 'double':
                for i in range(len(graph_data.inputs)):
                    graph_data.inputs[i] = graph_data.inputs[i].double()

        run_configs = get_run_configs(configs)

        for config_id, run_config in enumerate(run_configs):
            if 'config_id' in configs:
                # if config_id is provided in the config file, add it to the current config_id
                config_id += configs['config_id']
            # config_id to string with leading zeros
            c_id = f'Configuration_{str(config_id).zfill(6)}'
            run_configuration(c_id, run_config, graph_data, graph_db_name, run_id, validation_id, validation_number,
                              configs)
        # copy config file to the results directory if it is not already there
        if not os.path.exists(f"{r_path}{graph_db_name}/config.yml"):
            os.system(f"cp {absolute_path}{config} {r_path}{graph_db_name}/config.yml")
    else:
        #print that config file is not provided
        print("Please provide a configuration file")


def validation_step(run_id, validation_id, graph_data: GraphData.GraphData, para: Parameters.Parameters):
    """
    Split the data in training validation and test set
    """
    seed = 56874687 + validation_id + para.n_val_runs * run_id
    data = Load_Splits(para.splits_path, para.db)
    test_data = np.asarray(data[0][validation_id], dtype=int)
    training_data = np.asarray(data[1][validation_id], dtype=int)
    validate_data = np.asarray(data[2][validation_id], dtype=int)

    method = ModelEvaluation(run_id, validation_id, graph_data, training_data, validate_data, test_data, seed, para)

    """
    Run the method
    """
    method.Run()


def run_configuration(config_id, run_config, graph_data: GraphData, graph_db_name, run_id, validation_id,
                      validation_number,
                      configs):
    # get the data path from the config file
    data_path = configs['paths']['data']
    r_path = configs['paths']['results']
    l_path = configs['paths']['labels']
    properties_path = configs['paths']['properties']
    splits_path = configs['paths']['splits']
    # path do db and db
    results_path = r_path + graph_db_name + "/Results/"
    print_layer_init = True
    # if debug mode is on, turn on all print and draw options
    if configs['mode'] == "debug":
        draw = configs['additional_options']['draw']
        print_results = configs['additional_options']['print_results']
        save_prediction_values = configs['additional_options']['save_prediction_values']
        save_weights = configs['additional_options']['save_weights']
        plot_graphs = configs['additional_options']['plot_graphs']
    # if fast mode is on, turn off all print and draw options
    if configs['mode'] == "experiments":
        draw = False
        print_results = False
        save_weights = False
        save_prediction_values = False
        plot_graphs = False
        print_layer_init = False

    for i, l in enumerate(run_config.layers):
        # add the labels to the graph data
        label_path = f"{l_path}{graph_db_name}_{l.get_layer_string()}_labels.txt"
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[l.get_layer_string()] = g_labels
        elif l.layer_type == "combined":  # create combined file if it is a combined layer and the file does not exist
            combined_labels = []
            # get the labels for each layer in the combined layer
            for x in l.layer_dict['sub_labels']:
                sub_layer = Layer(x, i)
                sub_label_path = f"{l_path}/{graph_db_name}_{sub_layer.get_layer_string()}_labels.txt"
                if os.path.exists(sub_label_path):
                    g_labels = load_labels(path=sub_label_path)
                    combined_labels.append(g_labels)
                else:
                    # raise an error if the file does not exist
                    raise FileNotFoundError(f"File {sub_label_path} does not exist")
            # combine the labels and save them
            g_labels = combine_node_labels(combined_labels)
            graph_data.node_labels[l.get_layer_string()] = g_labels
            save_node_labels(data_path=f'{l_path}/', db_names=[graph_db_name], labels=g_labels.node_labels,
                             name=l.get_layer_string(), max_label_num=l.node_labels)
        else:
            # raise an error if the file does not exist and add the absolute path to the error message
            raise FileNotFoundError(f"File {label_path} does not exist")
        # add the properties to the graph data
        if 'properties' in l.layer_dict:
            prop_dict = l.layer_dict['properties']
            prop_name = prop_dict['name']
            if prop_name not in graph_data.properties:
                graph_data.properties[prop_name] = Properties(path=properties_path, db_name=graph_db_name,
                                                              property_name=prop_dict['name'],
                                                              valid_values=prop_dict['values'], layer_id=l.layer_id)
            else:
                graph_data.properties[prop_name].add_properties(prop_dict['values'], l.layer_id)
        pass

    para = Parameters.Parameters()

    """
        BenchmarkGraphs parameters
    """
    para.set_data_param(path=data_path, results_path=results_path,
                        splits_path=splits_path,
                        db=graph_db_name,
                        max_coding=1,
                        layers=run_config.layers,
                        batch_size=run_config.batch_size, node_features=1,
                        load_splits=configs['load_splits'],
                        configs=configs,
                        run_config=run_config, )

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_number, validation_id=validation_id,
                              config_id=config_id,
                              n_epochs=run_config.epochs,
                              learning_rate=run_config.lr, dropout=run_config.dropout,
                              balance_data=configs['balance_training'],
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
        if not os.path.exists(f"{r_path}{para.db}/Plots"):
            os.makedirs(f"{r_path}{para.db}/Plots")
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             f"{r_path}{para.db}/Plots/graph_{str(i).zfill(5)}.png")

    validation_step(para.run_id, para.validation_id, graph_data, para)


if __name__ == '__main__':
    main()
