'''
Created on 15.03.2019

@author: florian
'''
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from GraphData.DataSplits.load_splits import Load_Splits
from GraphData.Distances.load_distances import load_distances
from GraphData.Labels.generator.load_labels import load_labels
from Layers.GraphLayers import Layer
from LoadData.csl import CSL
from GraphData import GraphData, NodeLabeling
from Methods.GraphRuleMethod import GraphRuleMethod
from Parameters import Parameters
import ReadWriteGraphs.GraphDataToGraphList as gdtgl




class RunConfiguration():
    def __init__(self, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, loss):
        self.network_architecture = network_architecture
        self.layers = layers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss

    def print(self):
        print(f"Network architecture: {self.network_architecture}")
        print(f"Layers: {self.layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Dropout: {self.dropout}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Loss: {self.loss}")



@click.command()
@click.option('--graph_db_name', default="MUTAG", type=str, help='Database name')
@click.option('--run_id', default=0, type=int)
@click.option('--validation_number', default=10, type=int)
@click.option('--validation_id', default=0, type=int)
@click.option('--config', default=None, type=str)
# current configuration
#--graph_db_name NCI1 --config config.yml

def main(graph_db_name, run_id, validation_number, validation_id, config):
    if config is not None:
        # read the config yml file
        configs = yaml.safe_load(open(config))
        # get the data path from the config file
        data_path = configs['paths']['data']
        r_path = configs['paths']['results']
        distance_path = configs['paths']['distances']
        splits_path = configs['paths']['splits']

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
        if graph_db_name == "CSL":
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
            if os.path.isfile(f'{distance_path}{graph_db_name}_distances.pkl'):
                distance_list = load_distances(db_name=graph_db_name,
                                               path=f'{distance_path}{graph_db_name}_distances.pkl')
                graph_data.distance_list = distance_list
            # TODO: find other labeling method
        else:
            graph_data = GraphData.GraphData()
            graph_data.init_from_graph_db(data_path, graph_db_name, with_distances=True, with_cycles=False,
                                          relabel_nodes=True, use_features=configs['use_features'], use_attributes=configs['use_attributes'],
                                          distances_path=distance_path)

        # define the network type from the config file
        run_configs = []
        # iterate over all network architectures
        for network_architecture in configs['networks']:
            layers = []
            # get all different run configurations
            for l in network_architecture:
                layers.append(Layer(l))
            for b in configs['batch_size']:
                for lr in configs['learning_rate']:
                    for e in configs['epochs']:
                        for d in configs['dropout']:
                            for o in configs['optimizer']:
                                for loss in configs['loss']:
                                    run_configs.append(RunConfiguration(network_architecture, layers, b, lr, e, d, o, loss))

        # get the best configuration and run it
        config_id = get_best_configuration(graph_db_name, configs)

        c_id = f'Best_Configuration_{str(config_id).zfill(6)}'
        run_configuration(c_id, run_configs[config_id], graph_data, graph_db_name, run_id, validation_id, validation_number, configs)
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

    method = GraphRuleMethod(run_id, validation_id, graph_data, training_data, validate_data, test_data, seed, para)

    """
    Run the method
    """
    method.Run()


def run_configuration(config_id, run_config, graph_data, graph_db_name, run_id, validation_id, validation_number, configs):
    # get the data path from the config file
    data_path = configs['paths']['data']
    r_path = configs['paths']['results']
    distance_path = configs['paths']['distances']
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
    for l in run_config.layers:
        label_path = f"GraphData/Labels/{graph_db_name}_{l.get_layer_string()}_labels.txt"
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[l.get_layer_string()] = g_labels
        else:
            # raise an error if the file does not exist
            raise FileNotFoundError(f"File {label_path} does not exist")

    para = Parameters.Parameters()

    """
        Data parameters
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


def get_best_configuration(db_name, configs) -> int:
    evaluation = {}
    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    network_files = []
    for file in os.listdir(f"{configs['paths']['results']}/{db_name}/Results/"):
        if file.endswith(".txt") and file.find("Best") == -1:
            network_files.append(file)
        elif file.endswith(".csv") and -1 != file.find("run_id_0") and file.find("Best") == -1:
            files.append(file)

    # get the ids from the network files
    ids = []
    for file in network_files:
        ids.append(file.split("_")[-2])

    for id in ids:
        df_all = None
        for i, file in enumerate(files):
            if file.find(f"_{id}_") != -1:
                df = pd.read_csv(f"RESULTS/{db_name}/Results/{file}", delimiter=";")
                # concatenate the dataframes
                if df_all is None:
                    df_all = df
                else:
                    df_all = pd.concat([df_all, df], ignore_index=True)

        # group the data by RunNumberValidationNumber
        groups = df_all.groupby('ValidationNumber')

        indices = []
        # get the best validation accuracy for each validation run
        for name, group in groups:
            # get the maximum validation accuracy
            max_val_acc = group['ValidationAccuracy'].max()
            # get the row with the maximum validation accuracy
            max_row = group[group['ValidationAccuracy'] == max_val_acc]
            # get the minimum validation loss if column exists
            #if 'ValidationLoss' in group.columns:
            #    max_val_acc = group['ValidationLoss'].min()
            #    max_row = group[group['ValidationLoss'] == max_val_acc]

            # get row with the minimum validation loss
            min_val_loss = max_row['ValidationLoss'].min()
            max_row = group[group['ValidationLoss'] == min_val_loss]
            max_row = max_row.iloc[-1]
            # get the index of the row
            index = max_row.name
            indices.append(index)

        # get the rows with the indices
        df_validation = df_all.loc[indices]

        # get the average and deviation over all runs
        df_validation['EpochLoss'] *= df_validation['TrainingSize']
        df_validation['TestAccuracy'] *= df_validation['TestSize']
        df_validation['ValidationAccuracy'] *= df_validation['ValidationSize']
        df_validation['ValidationLoss'] *= df_validation['ValidationSize']
        avg = df_validation.mean(numeric_only=True)

        avg['EpochLoss'] /= avg['TrainingSize']
        avg['TestAccuracy'] /= avg['TestSize']
        avg['ValidationAccuracy'] /= avg['ValidationSize']
        avg['ValidationLoss'] /= avg['ValidationSize']

        std = df_validation.std(numeric_only=True)
        std['EpochLoss'] /= avg['TrainingSize']
        std['TestAccuracy'] /= avg['TestSize']
        std['ValidationAccuracy'] /= avg['ValidationSize']
        std['ValidationLoss'] /= avg['ValidationSize']

        evaluation[id] = [avg['TestAccuracy'], std['TestAccuracy'], avg['ValidationAccuracy'],
                              std['ValidationAccuracy'],
                              avg['ValidationLoss'], std['ValidationLoss']]
        # print evaluation
        print(f"Configuration {id}")
        print(f"Test Accuracy: {avg['TestAccuracy']} +- {std['TestAccuracy']}")
        print(f"Validation Accuracy: {avg['ValidationAccuracy']} +- {std['ValidationAccuracy']}")
        print(f"Validation Loss: {avg['ValidationLoss']} +- {std['ValidationLoss']}")


    # print the evaluation items with the k highest validation accuracies
    print(f"Top 5 Validation Accuracies for {db_name}")
    k = 5
    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][2], reverse=True)


    for i in range(min(k, len(sorted_evaluation))):
        sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][2], reverse=True)

    # print the id of the best configuration
    print(f"Best configuration: {sorted_evaluation[0][0]}")
    return int(sorted_evaluation[0][0])

if __name__ == '__main__':
    main()
