## Real World Data
import itertools
from pathlib import Path

import click
import numpy as np
import torch
import joblib

from src.Architectures.ShareGNN.Parameters import Parameters
from src.Experiment.ExperimentMain import ExperimentMain, preprocess_graph_data
from src.Experiment.ModelConfiguration import ModelConfiguration
from src.Experiment.RunConfiguration import get_run_configs
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.Preprocessing.load_preprocessed import load_preprocessed_data_and_parameters
from src.utils.load_splits import Load_Splits


def evaluate_gnn(num_threads=-1, db='MUTAG', path_strategy='i-E_d-IsoN', evaluation_folder='Evaluation'):
    # Load and preprocess the experiment
    experiment_base = ExperimentMain(Path(f'Examples/GED/Configs/main_config_{db}.yml'))
    experiment_base.ExperimentPreprocessing(num_threads=num_threads)
    experiment_paths = ExperimentMain(Path(f'Examples/GED/Configs/paths_config_{db}.yml'))
    experiment_paths.ExperimentPreprocessing(num_threads=num_threads)
    run_id = 0



    # evaluate the pretrained model on the original data only for testing
    for db_id, config in enumerate(experiment_base.network_configurations[db]):
        split_data = Load_Splits(config['paths']['splits'])
        train_ids = split_data['train']
        validation_ids = split_data['validation']
        test_ids = split_data['test']
        # get all possible hyperparameter configurations from the config files
        run_configs = get_run_configs(config)
        for config_id, run_config in enumerate(run_configs):
            for val_id in range(config['validation_folds']):
                model = experiment_base.load_ordinary_model(db_name=db, validation_id=val_id, best=False, run_id=run_id, experiment_db_id=db_id)
                model.eval()

                # create the model configuration object
                graph_data = preprocess_graph_data(config)
                path_graph_data = preprocess_graph_data(experiment_paths.network_configurations[f'{db}_{path_strategy}'][db_id])
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
                print(f"Evaluating model for config_id {config_id}, val_id {val_id} on training set")
                train_values, train_outputs = configuration.evaluate_network(graph_ids=train_ids[val_id], do_print=True, with_loss=True)
                # create Evaluation folder if it does not exist
                if not configuration.results_path.joinpath(evaluation_folder).exists():
                    configuration.results_path.joinpath(evaluation_folder).mkdir(parents=True, exist_ok=True)
                # create an empty torch tensor to save the outputs columns are graph_id, target_value, rest: output_values
                train_results_tensor = torch.empty((len(train_ids[val_id]), 2 + (train_outputs[0].shape[0] if isinstance(train_outputs[0], torch.Tensor) else 1)))
                for i, (gid, tval, oval) in enumerate(zip(train_ids[val_id],
                                                      train_values,
                                                      train_outputs)):
                    train_results_tensor[i, 0] = gid
                    train_results_tensor[i, 1] = tval
                    if isinstance(oval, torch.Tensor):
                        train_results_tensor[i, 2:] = oval
                    else:
                        train_results_tensor[i, 2] = oval
                torch.save(train_results_tensor, configuration.results_path.joinpath(evaluation_folder).joinpath(f'train_results_config{config_id}_val{val_id}_{db}.pt'))
                # save the results sorted by the graph ids as graph_id, target_value, output_value
                with open(configuration.results_path.joinpath(evaluation_folder).joinpath(f'train_results_config{config_id}_val{val_id}_{db}.txt'), 'w') as f:
                    f.write('graph_id\ttarget_value\toutput_value\n')
                    for gid, tval, oval in sorted(zip(train_ids[val_id],
                                                      train_values,
                                                      train_outputs), key=lambda x: x[0]):
                        #save ovals as space separated values if oval is a tensor
                        if isinstance(oval, torch.Tensor):
                            oval = ' '.join([str(v.item()) for v in oval])
                        f.write(f'{gid}\t{tval}\t{oval}\n')
                print(f"Evaluating model for config_id {config_id}, val_id {val_id} on validation set")
                validation_values, validation_outputs = configuration.evaluate_network(graph_ids=validation_ids[val_id], do_print=True, with_loss=True)

                # do the same for validation set
                # create an empty torch tensor to save the outputs columns are graph_id, target_value, rest: output_values
                validation_results_tensor = torch.empty((len(validation_ids[val_id]), 2 + (validation_outputs[0].shape[0] if isinstance(validation_outputs[0], torch.Tensor) else 1)))
                for i, (gid, tval, oval) in enumerate(zip(validation_ids[val_id],
                                                      validation_values,
                                                      validation_outputs)):
                    validation_results_tensor[i, 0] = gid
                    validation_results_tensor[i, 1] = tval
                    if isinstance(oval, torch.Tensor):
                        validation_results_tensor[i, 2:] = oval
                    else:
                        validation_results_tensor[i, 2] = oval
                torch.save(validation_results_tensor, configuration.results_path.joinpath(evaluation_folder).joinpath(f'validation_results_config{config_id}_val{val_id}_{db}.pt'))

                # save the results sorted by the graph ids as graph_id, target_value, output_value
                with open(configuration.results_path.joinpath(evaluation_folder).joinpath(f'validation_results_config{config_id}_val{val_id}_{db}.txt'), 'w') as f:
                    f.write('graph_id\ttarget_value\toutput_value\n')
                    for gid, tval, oval in sorted(zip(validation_ids[val_id],
                                                      validation_values,
                                                      validation_outputs), key=lambda x: x[0]):
                        #save ovals as space separated values if oval is a tensor
                        if isinstance(oval, torch.Tensor):
                            oval = ' '.join([str(v.item()) for v in oval])
                        f.write(f'{gid}\t{tval}\t{oval}\n')

                configuration_paths = ModelConfiguration(run_id, val_id, path_graph_data,
                                                   (train_ids, validation_ids, test_ids),
                                                   seed, para)

                # Initialize the graph neural network
                configuration_paths.initialize_model(pretrained_network=model,
                                               use_model=configuration_paths.para.run_config.config.get('use_model',
                                                                                                  'ShareGNN'))
                print(f"Evaluating PATHS model for config_id {config_id}, val_id {val_id} on path set")
                target_values, target_outputs = configuration_paths.evaluate_network(graph_ids=list(range(len(path_graph_data))))

                # do the same with path target outputs using columns source_id, step_id, target_id, operation_id, rest: output_values
                # operation_id: is a number -1 for source graph, 0 for first NODE INSERTION, 1 for NODE DELETION, 2 for NODE RELABEL, 3 for EDGE INSERTION, 4 for EDGE DELETION and 5 for EDGE RELABEL
                path_results_tensor = torch.empty((len(path_graph_data), 4 + (target_outputs[0].shape[0] if isinstance(target_outputs[0], torch.Tensor) else 1)))
                skip = 0
                # save the path target values using the format source_id, step_id, target_value operation by loading the path info
                operation_information = []
                with open(Path(experiment_paths.main_config['paths']['data']).joinpath(f'{db}_{path_strategy}').joinpath(f'{db}_edit_paths_data.txt'), 'r') as f:
                    for line in f.readlines():
                        operation_information.append(line)
                for i, operation_line in enumerate(operation_information):
                    if i + skip >= len(target_outputs):
                        break
                    target_output = target_outputs[i + skip]
                    operation_line = operation_line.strip()
                    if operation_line.startswith('#') or operation_line == '':
                        continue
                    parts = operation_line.split(' ')
                    source_id = parts[0]
                    step_id = parts[1]
                    target_id = parts[2]
                    operation_element = parts[3]
                    operation_value = parts[4]
                    operation_type = parts[5]
                    # if the step_id is 0 write an additional line evaluating on the source graph
                    if step_id == '0':
                        path_results_tensor[i + skip, 0] = int(source_id)
                        path_results_tensor[i + skip, 1] = -1  # step_id -1 for source graph
                        path_results_tensor[i + skip, 2] = int(target_id)
                        path_results_tensor[i + skip, 3] = -1  # operation_id -1 for source graph
                        if isinstance(target_output, torch.Tensor):
                            path_results_tensor[i + skip, 4:] = target_output
                        else:
                            path_results_tensor[i + skip, 4] = target_output
                        skip += 1
                        if i + skip >= len(target_outputs):
                            break
                        target_output = target_outputs[i + skip]
                    operation_id = {'NODE_INSERT': 0,
                                    'NODE_DELETE': 1,
                                    'NODE_RELABEL': 2,
                                    'EDGE_INSERT': 3,
                                    'EDGE_DELETE': 4,
                                    'EDGE_RELABEL': 5}.get(f'{operation_element}_{operation_type}', -2)
                    path_results_tensor[i + skip, 0] = int(source_id)
                    path_results_tensor[i + skip, 1] = int(step_id)
                    path_results_tensor[i + skip, 2] = int(target_id)
                    path_results_tensor[i + skip, 3] = operation_id
                    if isinstance(target_output, torch.Tensor):
                        path_results_tensor[i + skip, 4:] = target_output
                    else:
                        path_results_tensor[i + skip, 4] = target_output
                    if operation_id == -2:
                        raise ValueError(f'Unknown operation type: {operation_element} {operation_type}')
                torch.save(path_results_tensor, configuration_paths.results_path.joinpath(evaluation_folder).joinpath(f'path_results_config{config_id}_val{val_id}_{db}_{path_strategy}.pt'))



                # create evaluation folder if it does not exist
                if not configuration_paths.results_path.joinpath(evaluation_folder).exists():
                    configuration_paths.results_path.joinpath(evaluation_folder).mkdir(parents=True, exist_ok=True)
                with open(configuration_paths.results_path.joinpath(evaluation_folder).joinpath(f'path_results_config{config_id}_val{val_id}_{db}_{path_strategy}.txt'), 'w') as f:
                    f.write('source_id\tstep_id\ttarget_id\toperation\ttarget_value\n')
                    skip = 0
                    for i, operation_line in enumerate(operation_information):
                        if i + skip >= len(target_outputs):
                            break
                        target_output = target_outputs[i + skip]
                        if isinstance(target_output, torch.Tensor):
                            target_output = ' '.join([str(v.item()) for v in target_output])
                        operation_line = operation_line.strip()
                        if operation_line.startswith('#') or operation_line == '':
                            continue
                        parts = operation_line.split(' ')
                        source_id = parts[0]
                        step_id = parts[1]
                        target_id = parts[2]
                        operation_element = parts[3]
                        operation_value = parts[4]
                        operation_type = parts[5]
                        # if the step_id is 0 write an additional line evaluating on the source graph
                        if step_id == '0':
                            f.write(f'{source_id}\tS\t{target_id}\t{target_output}\tNONE\n')
                            skip += 1
                            if i + skip >= len(target_outputs):
                                break
                            target_output = target_outputs[i + skip]
                            if isinstance(target_output, torch.Tensor):
                                target_output = ' '.join([str(v.item()) for v in target_output])
                        f.write(f'{source_id}\t{step_id}\t{target_id}\t{target_output}\t{operation_element} {operation_value} {operation_type}\n')
                    # after finishing all lines check if there are remaining target values to write
                pass


def main():

    # dataset
    dbs = ['MUTAG', 'Mutagenicity', 'NCI1', 'DHFR', 'NCI109']
    path_strategies = ['Rnd', 'i-E_d-IsoN']

    dbs = ['Mutagenicity']
    evaluation_folder = 'Evaluation'
    tasks = list(itertools.product(dbs, path_strategies))
    # parallel evaluation over datasets and path strategies
    joblib.Parallel(n_jobs=len(tasks))(
        joblib.delayed(evaluate_gnn)(db=db, path_strategy=path_strategy, evaluation_folder=evaluation_folder)
        for db, path_strategy in tasks
    )



if __name__ == '__main__':
    main()