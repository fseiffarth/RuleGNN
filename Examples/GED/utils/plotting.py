from pathlib import Path

import torch

from src.Architectures.ShareGNN.Parameters import Parameters
from src.Experiment.ExperimentMain import ExperimentMain, preprocess_graph_data
from src.Experiment.ModelConfiguration import ModelConfiguration
from src.Experiment.RunConfiguration import get_run_configs
from src.Preprocessing.load_preprocessed import load_preprocessed_data_and_parameters
from src.utils.load_splits import Load_Splits


def table_accuracies(dbs, gnn_algos):
    # dict of key: (db, gnn_algorithm), value: dict of config_id -> (mean_validation_accuracy, std_validation_accuracy)
    results = {}
    run_id = 0
    for gnn_algorithm in gnn_algos:
        for db in dbs:
            # Load and preprocess the experiment
            experiment_base = ExperimentMain(Path(f'Examples/GED/Configs/main_config_{db}.yml'))
            experiment_base.ExperimentPreprocessing(num_threads=-1)
            # evaluate the pretrained model on the original data only for testing
            for db_id, config in enumerate(experiment_base.network_configurations[db]):
                algorithm = config['network_config_file'].split('_')[1]
                # also remove possible .yml at the end
                algorithm = algorithm.replace('.yml', '')
                if gnn_algorithm is not None and algorithm != gnn_algorithm:
                    continue
                split_data = Load_Splits(config['paths']['splits'])
                train_ids = split_data['train']
                validation_ids = split_data['validation']
                test_ids = split_data['test']

                # get the graph data
                graph_data = preprocess_graph_data(config)

                # get all possible hyperparameter configurations from the config files
                run_configs = get_run_configs(config)
                for config_id, run_config in enumerate(run_configs):
                    validation_results = {}
                    for val_id in range(config['validation_folds']):
                        model = experiment_base.load_ordinary_model(db_name=db, validation_id=val_id, best=False, run_id=run_id, experiment_db_id=db_id)
                        model.eval()



                        para = Parameters()
                        load_preprocessed_data_and_parameters(config_id=config_id,
                                                              run_id=run_id,
                                                              validation_id=val_id,
                                                              validation_folds=config.get('validation_folds', 10),
                                                              graph_data=graph_data, run_config=run_config, para=para)
                        seed = 42 + val_id + para.n_val_runs * run_id
                        configuration = ModelConfiguration(run_id, val_id, graph_data, (train_ids, validation_ids, test_ids),
                                                           seed, para)
                        # Initialize the graph neural network
                        configuration.initialize_model(pretrained_network=model,
                                                       use_model=configuration.para.run_config.config.get('use_model',
                                                                                                          'ShareGNN'))
                        print(f"Evaluating model for config_id {config_id}, val_id {val_id} on validation set")
                        validation_values, validation_outputs = configuration.evaluate_network(graph_ids=validation_ids[val_id], do_print=True, with_loss=True)
                        predictions = torch.argmax(validation_outputs, dim=1)
                        validation_accuracy = 100 * torch.sum(predictions == validation_values).item() / len(validation_values)
                        validation_results[val_id] = validation_accuracy
                    # store the mean and std of the validation accuracies
                    results[(db, gnn_algorithm, config_id)] = validation_results
    # Print the results in a table format (columns dbs, rows gnn_algos) with mean and std of validation accuracies (config_id with highest mean accuracy)
    for db in dbs:
        print(f"Results for dataset {db}:")
        print("GNN Algorithm\tMean Validation Accuracy (%)\tStd Validation Accuracy (%)")
        for gnn_algorithm in gnn_algos:
            best_mean_accuracy = 0
            best_std_accuracy = 0
            for config_id in range(len(run_configs)):
                if (db, gnn_algorithm, config_id) in results:
                    accuracies = [results[(db, gnn_algorithm, config_id)][val_id] for val_id in range(config['validation_folds'])]
                    mean_accuracy = sum(accuracies) / len(accuracies)
                    std_accuracy = (sum((x - mean_accuracy) ** 2 for x in accuracies) / len(accuracies)) ** 0.5
                    if mean_accuracy > best_mean_accuracy:
                        best_mean_accuracy = mean_accuracy
                        best_std_accuracy = std_accuracy
            print(f"{gnn_algorithm}\t{best_mean_accuracy:.2f}\t{best_std_accuracy:.2f}")
        print("\n")



def main():
    dbs = ['MUTAG', 'Mutagenicity', 'NCI1', 'DHFR', 'NCI109', ]
    gnn_algos = ['GCN', 'GATv2', 'GraphSAGE', 'GIN']

    table_accuracies(dbs, gnn_algos)


if __name__ == '__main__':
    main()