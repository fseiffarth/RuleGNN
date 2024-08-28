'''
Created on 15.03.2019

@author:
'''
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from scripts.find_best_models import run_configuration
from src.utils.GraphData import get_graph_data
from src.utils.RunConfiguration import get_run_configs


def run_best_models(graph_db_name, run_id, validation_number, validation_id, graph_format, transfer, config, evaluation_type):
    if config is not None:
        absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        absolute_path = Path(absolute_path)
        config_path = absolute_path.joinpath(config)
        # read the config yml file
        configs = yaml.safe_load(open(config_path))
        # get the data path from the config file
        # add absolute path to the config data paths
        configs['paths']['data'] = absolute_path.joinpath(configs['paths']['data'])
        configs['paths']['results'] = absolute_path.joinpath(configs['paths']['results'])
        configs['paths']['labels'] = absolute_path.joinpath(configs['paths']['labels'])
        configs['paths']['properties'] = absolute_path.joinpath(configs['paths']['properties'])
        configs['paths']['splits'] = absolute_path.joinpath(configs['paths']['splits'])
        configs['format'] = graph_format
        configs['transfer'] = transfer
        # set best model to true
        configs['best_model'] = True

        data_path = configs['paths']['data']
        r_path = configs['paths']['results']

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
        if not os.path.exists(r_path.joinpath(graph_db_name).joinpath("Results")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name).joinpath("Results"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name).joinpath("Plots")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name).joinpath("Plots"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name).joinpath("Weights")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name).joinpath("Weights"))
            except:
                pass
        if not os.path.exists(r_path.joinpath(graph_db_name).joinpath("Models")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name).joinpath("Models"))
            except:
                pass

        plt.ion()

        """
        Create Input data, information and labels from the graphs for training and testing
        """
        graph_data = get_graph_data(db_name=graph_db_name, data_path=data_path, use_features=configs['use_features'],
                                    use_attributes=configs['use_attributes'], data_format=graph_format)
        # adapt the precision of the input data
        if 'precision' in configs:
            if configs['precision'] == 'double':
                for i in range(len(graph_data.inputs)):
                    graph_data.inputs[i] = graph_data.inputs[i].double()

        run_configs = get_run_configs(configs)
        # get the best configuration and run it
        config_id = get_best_configuration(graph_db_name, configs, type=evaluation_type)

        c_id = f'Best_Configuration_{str(config_id).zfill(6)}'
        run_configuration(c_id, run_configs[config_id], graph_data, graph_db_name, run_id, validation_id, validation_number, configs)
    else:
        #print that config file is not provided
        print("Please provide a configuration file")

def get_best_configuration(db_name, configs, type='loss') -> int:
    evaluation = {}
    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    network_files = []
    search_path = configs['paths']['results'].joinpath(db_name).joinpath("Results")
    for file in os.listdir(search_path):
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
                csv_path = configs['paths']['results'].joinpath(db_name).joinpath("Results").joinpath(file)
                df = pd.read_csv(csv_path, delimiter=";")
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
            if type == 'accuracy':
                # get the maximum validation accuracy
                max_val_acc = group['ValidationAccuracy'].max()
                # get the row with the maximum validation accuracy
                max_row = group[group['ValidationAccuracy'] == max_val_acc]
            elif type == 'loss':
                # get the minimum validation loss if column exists
                if 'ValidationLoss' in group.columns:
                    max_val_acc = group['ValidationLoss'].min()
                    max_row = group[group['ValidationLoss'] == max_val_acc]

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
    if type == 'accuracy':
        sort_key = 2
        reversed_sort = True
    elif type == 'loss':
        sort_key = 4
        reversed_sort = False
    sorted_evaluation = sorted(evaluation.items(), key=lambda x: x[1][sort_key], reverse=reversed_sort)


    for i in range(min(k, len(sorted_evaluation))):
        sorted_evaluation = sorted(sorted_evaluation, key=lambda x: x[1][sort_key], reverse=reversed_sort)

    # print the id of the best configuration
    print(f"Best configuration: {sorted_evaluation[0][0]}")
    return int(sorted_evaluation[0][0])

if __name__ == '__main__':
    run_best_models()
