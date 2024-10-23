import os
from pathlib import Path

import joblib
import numpy as np
import torch
import yaml

from scripts.Evaluation.EvaluationFinal import model_selection_evaluation
from scripts.Preprocessing import Preprocessing
import src.utils.SyntheticGraphs as synthetic_graphs
from src.Architectures.RuleGNN import RuleGNN
from src.Methods.ModelEvaluation import ModelEvaluation
from src.Preprocessing.load_preprocessed import load_preprocessed_data_and_parameters
from src.utils.GraphData import get_graph_data
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_splits import Load_Splits
from src.utils.path_conversions import config_paths_to_absolute




class ExperimentMain:
    """
    This is the main class to run RuleGNN experiments.
    All experiment parameters are defined in the main config file and the experiment config file.
    """
    def __init__(self, main_config_path: os.path):
        self.main_config_path = main_config_path
        if not os.path.exists(main_config_path):
            raise FileNotFoundError(f"Config file {main_config_path} not found")
        try:
            self.main_config = yaml.safe_load(open(main_config_path))
        except:
            raise ValueError(f"Config file {main_config_path} could not be loaded")
        self.experiment_configurations = {}
        self.dataset_configs = {}
        for dataset in self.main_config['datasets']:
            experiment_configuration = self.update_experiment_configuration(dataset)
            experiment_configuration['format'] = 'NEL'
            self.experiment_configurations[dataset['name']] = experiment_configuration
            self.dataset_configs[dataset['name']] = dataset



        self.check_config_consistency()


    def GridSearch(self):
        """
        Run over all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        """
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            self.create_folders(dataset['name'])
            print(f"Running experiment for dataset {dataset['name']}")
            experiment_configuration = self.experiment_configurations[dataset['name']]
            # determine the number of parallel jobs
            num_workers = experiment_configuration.get('num_workers', min(os.cpu_count(), dataset.get('validation_folds', 10)))
            graph_data = preprocess_graph_data(dataset['name'], experiment_configuration)

            # copy config file to the results directory if it is not already there
            absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            absolute_path = Path(absolute_path)
            copy_experiment_config(absolute_path, experiment_configuration,
                                   dataset.get('experiment_config_file', ''),
                                   dataset['name'])

            run_configs = get_run_configs(experiment_configuration)
            config_id_names = {}
            for idx, run_config in enumerate(run_configs):
                config_id = idx + experiment_configuration.get('config_id', 0)
                # config_id to string with leading zeros
                config_id_names[idx] = f'Configuration_{str(config_id).zfill(6)}'
            # print the number of run configurations to be tested
            print(f"Number of hyperparameter configurations: {len(run_configs)}")

            # zip validation_id,  config_id and run_id to parallelize over them
            run_loops = [(validation_id, run_id, c_idx) for validation_id in range(dataset.get('validation_folds', 10)) for run_id in range(dataset.get('num_runs', 1)) for c_idx in range(len(run_configs))]
            num_workers = min(num_workers, len(run_loops))
            print(f"Run the grid search for dataset {dataset['name']} using {dataset.get('validation_folds', 10)}-fold cross-validation and {num_workers} number of parallel jobs")
            joblib.Parallel(n_jobs=num_workers)(
                joblib.delayed(self.run_models)(dataset=dataset,
                                                graph_data=graph_data,
                                                run_config=run_configs[run_loops[i][2]],
                                                validation_id=run_loops[i][0],
                                                run_id=run_loops[i][1],
                                                config_id=config_id_names[run_loops[i][2]]) for i in range(len(run_loops)))

    def EvaluateResults(self, evaluate_best_model=False, evaluate_validation_only=False):
        """
        Evaluate the results of the experiments for all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        parameters:
        - dataset_names: list of strings with the names of the datasets to evaluate
        - evaluation_type: string with the type of evaluation to perform. Default is 'accuracy'. Other options are 'loss'. For accuracy, take the best model according to the validation accuracy. For loss, take the best model according to the validation loss.
        - evaluate_best_model: boolean to evaluate the best model of the experiment. Default is False. If False, evaluate the results of the experiment.
        - evaluate_validation_only: boolean to evaluate only on the validation set. Returns the epoch with the best validation accuracy/loss. Default is False.
        """
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            if evaluate_best_model:
                print(f"Evaluate the best model of the experiment for dataset {dataset['name']}")
            else:
                print(f"Evaluate the results of the experiment for dataset {dataset['name']}")

            experiment_configuration = self.update_experiment_configuration(dataset)

            model_selection_evaluation(db_name = dataset['name'],
                                       evaluate_best_model=evaluate_best_model,
                                       experiment_config=experiment_configuration,
                                       evaluate_validation_only=evaluate_validation_only)

    def RunBestModel(self):
        """
        Run over all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        """
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            print(f"Running experiment for dataset {dataset['name']}")
            validation_folds = dataset['validation_folds']

            # load the config file
            # run the best models
            # parallelize over (run_id, validation_id) pairs
            evaluation_run_number = self.main_config.get('evaluation_run_number', 3)
            experiment_configuration = self.update_experiment_configuration(dataset)
            parallelization_pairs = [(run_id, validation_id) for run_id in range(evaluation_run_number) for validation_id in range(validation_folds)]
            num_workers = experiment_configuration.get('num_workers', min(os.cpu_count(), len(parallelization_pairs)))
            graph_data = preprocess_graph_data(dataset['name'], experiment_configuration)
            best_config_id = None
            experiment_configuration['best_model'] = True
            # get the best configuration and run it
            best_config_id = model_selection_evaluation(db_name=dataset['name'],
                                                        get_best_model=True,
                                                        experiment_config=experiment_configuration)
            run_configs = get_run_configs(experiment_configuration)
            config_id = f'Best_Configuration_{str(best_config_id).zfill(6)}'
            print(f"Run the best model of dataset {dataset['name']} using {evaluation_run_number} different runs. The number of parallel jobs is {num_workers}")
            joblib.Parallel(n_jobs=num_workers)(joblib.delayed(self.run_models)(dataset=dataset,
                                                        graph_data=graph_data,
                                                        run_config=run_configs[best_config_id],
                                                        validation_id=validation_id,
                                                        run_id=run_id,
                                                        config_id=config_id)
                                             for run_id, validation_id in parallelization_pairs)

    def update_experiment_configuration(self, dataset):
        experiment_configuration_path = dataset.get('experiment_config_file', '')
        # load the config file
        experiment_configuration = yaml.load(open(experiment_configuration_path), Loader=yaml.FullLoader)
        paths = collect_paths(main_configuration=self.main_config, dataset_configuration=dataset,
                              experiment_configuration=experiment_configuration)
        experiment_configuration['paths'] = paths
        # paths to Path objects
        config_paths_to_absolute(experiment_configuration,
                                 Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))
        # update the experiment configuration with the global keys
        for key in self.main_config:
            if key not in experiment_configuration and key != 'datasets':
                experiment_configuration[key] = self.main_config[key]
        return experiment_configuration

    def Preprocess(self, num_jobs=-1):
        # parallelize over the datasets
        if num_jobs == -1:
            num_jobs = min(len(self.main_config['datasets']), os.cpu_count())
            num_jobs = self.main_config.get('num_workers', num_jobs)
        joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(self.PreprocessParallel)(dataset_configuration) for dataset_configuration in self.main_config['datasets'])

    def PreprocessParallel(self, dataset_configuration):
        db_name = dataset_configuration['name']
        experiment_config_file = dataset_configuration['experiment_config_file']
        experiment_configuration = yaml.load(open(experiment_config_file), Loader=yaml.FullLoader)
        data_generation = None
        data_generation_args = None
        if 'type' in dataset_configuration:
            if dataset_configuration['type'] == 'generate_from_function':
                if 'generate_function' in dataset_configuration:
                    data_generation = getattr(synthetic_graphs, dataset_configuration['generate_function'])
                    if 'generate_function_args' in dataset_configuration:
                        data_generation_args = dataset_configuration['generate_function_args']
                    else:
                        data_generation_args = None
                else:
                    data_generation = None
                    data_generation_args = None
            elif dataset_configuration['type'] == 'TUDataset':
                data_generation = 'TUDataset'
                data_generation_args = None
            else:
                print(
                    f"The type {dataset_configuration['type']} is not supported. Please use 'generate_from_function' or 'TUDataset'")

        if 'with_splits' in dataset_configuration:
            with_splits = dataset_configuration['with_splits']
        else:
            with_splits = True

        paths = collect_paths(main_configuration=self.main_config, dataset_configuration=dataset_configuration, experiment_configuration=experiment_configuration)
        experiment_configuration['paths'] = paths
        # paths to Path objects
        config_paths_to_absolute(experiment_configuration, Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))
        # preprocess the data
        Preprocessing(db_name=db_name, dataset_configuration=dataset_configuration, experiment_configuration=experiment_configuration, with_splits=with_splits,data_generation=data_generation, data_generation_args=data_generation_args)

    def check_config_consistency(self):
        pass

    def create_folders(self, graph_db_name):
        experiment_configuration = self.experiment_configurations[graph_db_name]
        r_path = experiment_configuration['paths']['results']
        # if not exists create the results directory
        if not os.path.exists(r_path):
            try:
                os.makedirs(r_path)
            except:
                raise FileNotFoundError(f"Results directory {r_path} not found")
        # if not exists create the directory for db under Results
        if not os.path.exists(r_path.joinpath(graph_db_name)):
            try:
                os.makedirs(r_path.joinpath(graph_db_name))
            except:
                raise FileNotFoundError(f"Results directory {r_path.joinpath(graph_db_name)} not found")
        # if not exists create the directory Results, Plots, Weights and Models under db
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Results")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Results"))
            except:
                raise FileNotFoundError(f"Results directory {r_path.joinpath(graph_db_name + '/Results')} not found")
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Plots")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Plots"))
            except:
                raise FileNotFoundError(f"Results directory {r_path.joinpath(graph_db_name + '/Plots')} not found")
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Weights")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Weights"))
            except:
                raise FileNotFoundError(f"Results directory {r_path.joinpath(graph_db_name + '/Weights')} not found")
        if not os.path.exists(r_path.joinpath(graph_db_name + "/Models")):
            try:
                os.makedirs(r_path.joinpath(graph_db_name + "/Models"))
            except:
                raise FileNotFoundError(f"Results directory {r_path.joinpath(graph_db_name + '/Models')} not found")

    def run_models(self, dataset, graph_data, run_config, validation_id=0, run_id=0, config_id=None):
        # print the current configuration
        print(f"Run the model for dataset {dataset['name']} with config_id {config_id}, run_id {run_id} and validation_id {validation_id}")
        para = Parameters()
        load_preprocessed_data_and_parameters(config_id=config_id,
                                              run_id=run_id,
                                              validation_id=validation_id,
                                              validation_folds=dataset.get('validation_folds', 10),
                                              graph_data=graph_data, run_config=run_config, para=para)
        """
        Split the data in training validation and test set
        """
        seed = 56874687 + validation_id + para.n_val_runs * run_id
        data = Load_Splits(para.splits_path, para.db, para.run_config.config.get('transfer', False))
        test_data = data[0][validation_id]
        train_data = data[1][validation_id]
        validation_data = data[2][validation_id]
        model_data = (np.array(train_data), np.array(validation_data), np.array(test_data))
        method = ModelEvaluation(run_id, validation_id, graph_data, model_data, seed, para)

        """
        Run the method
        """
        method.Run()

    def load_model(self, db_name, config_id=0, run_id=0, validation_id=0, best=True):
        experiment_configuration = self.experiment_configurations[db_name]
        graph_data = preprocess_graph_data(db_name, experiment_configuration)
        run_configs = get_run_configs(experiment_configuration)
        model_path = experiment_configuration['paths']['results'].joinpath(db_name).joinpath('Models')
        if best:
            # get config id of the best model
            if model_path.exists():
                # get one file from the directory
                file = next(model_path.iterdir())
                # get the config id from the file name
                config_id = int(file.name.split('_')[3])
            else:
                raise FileNotFoundError(f"Model directory {model_path} not found")
        run_config = run_configs[config_id]
        model_path = model_path.joinpath(f'model_Best_Configuration_{str(config_id).zfill(6)}_run_{run_id}_val_step_{validation_id}.pt')
        # check if the model exists
        if model_path.exists():
            with open(model_path, 'r'):
                para = Parameters()
                load_preprocessed_data_and_parameters(config_id=config_id,
                                                      run_id=run_id,
                                                      validation_id=validation_id,
                                                      graph_data=graph_data,
                                                      run_config=run_config,
                                                      para=para,
                                                      validation_folds=experiment_configuration.get('validation_folds', 10))

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)
                net = RuleGNN.RuleGNN(graph_data=graph_data,
                                      para=para,
                                      seed=0, device=run_config.config.get('device', 'cpu'))

                net.load_state_dict(torch.load(model_path, weights_only=True))
            return net
        else:
            raise FileNotFoundError(f"Model {model_path} not found")

    def evaluate_model(self, db_name, config_id=0, run_id=0, validation_id=0):
        # evaluate the performance of the model on the test data
        net = self.load_model(db_name, config_id, run_id, validation_id)
        experiment_configuration = self.experiment_configurations[db_name]
        graph_data = preprocess_graph_data(db_name, experiment_configuration)
        split_data = Load_Splits(experiment_configuration['paths']['splits'], db_name)
        test_data = np.asarray(split_data[0][validation_id], dtype=int)
        outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
        with torch.no_grad():
            for j, data_pos in enumerate(test_data, 0):
                inputs = torch.DoubleTensor(graph_data.input_data[data_pos])
                outputs[j] = net(inputs, data_pos)
            labels = graph_data.output_data[test_data]
            # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
            counter = 0
            correct = 0
            for i, x in enumerate(outputs, 0):
                if torch.argmax(x) == torch.argmax(labels[i]):
                    correct += 1
                counter += 1
            accuracy = correct / counter
            print(f"Dataset: {db_name}, Run Id: {run_id}, Validation Split Id: {validation_id}, Accuracy: {accuracy}")


def collect_paths(main_configuration, experiment_configuration, dataset_configuration=None):
    # first look into the main config file
    paths = main_configuration.get('paths', {})
    # if available add the data path from the dataset config file
    if dataset_configuration is not None and dataset_configuration.get('data', None) is not None:
        paths['data'] = dataset_configuration['data']

    # if there are paths in the experiment config file, overwrite the paths
    if experiment_configuration.get('paths', None) is not None:
        if experiment_configuration['paths'].get('data', None) is not None:
            paths['data'] = experiment_configuration['paths']['data']
        if experiment_configuration['paths'].get('results', None) is not None:
            paths['results'] = experiment_configuration['paths']['results']
        if experiment_configuration['paths'].get('splits', None) is not None:
            paths['splits'] = experiment_configuration['paths']['splits']
        if experiment_configuration['paths'].get('properties', None) is not None:
            paths['properties'] = experiment_configuration['paths']['properties']
        if experiment_configuration['paths'].get('labels', None) is not None:
            paths['labels'] = experiment_configuration['paths']['labels']

    # check wheter one of the paths is missing
    if 'data' not in paths:
        raise FileNotFoundError("Data path is missing")
    if 'results' not in paths:
        raise FileNotFoundError("Results path is missing")
    if 'splits' not in paths:
        raise FileNotFoundError("Splits path is missing")
    if 'properties' not in paths:
        raise FileNotFoundError("Properties path is missing")
    if 'labels' not in paths:
        raise FileNotFoundError("Labels path is missing")

    return paths

def copy_experiment_config(absolute_path, experiment_configuration, experiment_configuration_path,
                           graph_db_name):
    if not os.path.exists(experiment_configuration['paths']['results'].joinpath(f"{graph_db_name}/config.yml")):
        source_path = Path(absolute_path).joinpath(experiment_configuration_path)
        destination_path = experiment_configuration['paths']['results'].joinpath(f"{graph_db_name}/config.yml")
        # copy the config file to the results directory
        # if linux
        if os.name == 'posix':
            os.system(f"cp {source_path} {destination_path}")
        # if windows
        elif os.name == 'nt':
            os.system(f"copy {source_path} {destination_path}")


def preprocess_graph_data(db_name, experiment_configuration):
    """
            Create Input data, information and labels from the graphs for training and testing
            """
    graph_data = get_graph_data(db_name=db_name, data_path=experiment_configuration['paths']['data'],
                                task=experiment_configuration.get('task', 'graph_classification'),
                                input_features=experiment_configuration.get('input_features', None),
                                output_features=experiment_configuration.get('output_features', None),
                                graph_format=experiment_configuration.get('format', 'NEL'))
    graph_data.set_precision(experiment_configuration.get('precision', 'double'))
    return graph_data


