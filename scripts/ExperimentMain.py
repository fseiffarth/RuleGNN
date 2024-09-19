import os
from pathlib import Path

import joblib
import numpy as np
import yaml
from matplotlib import pyplot as plt

from scripts.Evaluation.EvaluationFinal import model_selection_evaluation
from scripts.Preprocessing import Preprocessing
import src.utils.SyntheticGraphs as synthetic_graphs
from src.Methods.ModelEvaluation import ModelEvaluation
from src.Preprocessing.load_preprocessed import load_preprocessed_data_and_parameters
from src.utils.GraphData import get_graph_data, GraphData
from src.utils.Parameters.Parameters import Parameters
from src.utils.RunConfiguration import get_run_configs
from src.utils.load_splits import Load_Splits
from src.utils.path_conversions import config_paths_to_absolute


class ExperimentMain:
    '''
    This is the main class to run the experiments for RuleGNNs. It reads the main config file which defines the datasets and the corresponding experiment config files.
    '''
    def __init__(self, main_config_path: os.path):
        self.main_config_path = main_config_path
        try:
            self.main_config = yaml.safe_load(open(main_config_path))
        except:
            raise FileNotFoundError(f"Config file {main_config_path} not found")
        self.check_config_consistency()
        self.dataset_names = [dataset['name'] for dataset in self.main_config['datasets']]


    def GridSearch(self, dataset_names=None):
        '''
        Run over all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        '''
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        if dataset_names is not None:
            self.dataset_names = dataset_names
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            print(f"Running experiment for dataset {dataset['name']}")
            validation_folds = dataset.get('validation_folds', 10)
            num_runs = dataset.get('num_runs', 1)

            experiment_configuration_path = dataset.get('experiment_config_file', '')
            # load the config file
            experiment_configuration = yaml.load(open(experiment_configuration_path), Loader=yaml.FullLoader)
            paths = collect_paths(main_configuration=self.main_config, dataset_configuration=dataset,
                          experiment_configuration=experiment_configuration)
            experiment_configuration['paths'] = paths
            # paths to Path objects
            config_paths_to_absolute(experiment_configuration, Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))
            # update the experiment configuration with the global keys
            for key in self.main_config:
                if key not in experiment_configuration and key != 'datasets':
                    experiment_configuration[key] = self.main_config[key]


            for run_id in range(num_runs):
                print(f"Run {run_id + 1} of {num_runs}")
                # find the best model hyperparameters using grid search and cross-validation
                print(f"Find the best hyperparameters for dataset {dataset['name']} using {validation_folds}-fold cross-validation and {validation_folds} number of parallel jobs")
                joblib.Parallel(n_jobs=validation_folds)(
                    joblib.delayed(self.find_best_models)(graph_db_name=dataset['name'],
                                                          validation_folds=validation_folds,
                                                          validation_id=validation_id, graph_format="NEL",
                                                          experiment_configuration=experiment_configuration, experiment_configuration_path=experiment_configuration_path, run_id=run_id) for validation_id in range(validation_folds))

    def EvaluateResults(self, dataset_names=None, evaluation_type='accuracy', evaluate_best_model=False, evaluate_validation_only=False):
        '''
        Evaluate the results of the experiments for all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        parameters:
        - dataset_names: list of strings with the names of the datasets to evaluate
        - evaluation_type: string with the type of evaluation to perform. Default is 'accuracy'. Other options are 'loss'. For accuracy, take the best model according to the validation accuracy. For loss, take the best model according to the validation loss.
        - evaluate_best_model: boolean to evaluate the best model of the experiment. Default is False. If False, evaluate the results of the experiment.
        - evaluate_validation_only: boolean to evaluate only on the validation set. Returns the epoch with the best validation accuracy/loss. Default is False.
        '''
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        if dataset_names is not None:
            self.dataset_names = dataset_names
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            if evaluate_best_model:
                print(f"Evaluate the best model of the experiment for dataset {dataset['name']}")
            else:
                print(f"Evaluate the results of the experiment for dataset {dataset['name']}")

            experiment_configuration_path = dataset.get('experiment_config_file', '')
            # load the config file
            experiment_configuration = yaml.load(open(experiment_configuration_path), Loader=yaml.FullLoader)
            paths = collect_paths(main_configuration=self.main_config, dataset_configuration=dataset,
                          experiment_configuration=experiment_configuration)
            experiment_configuration['paths'] = paths
            # paths to Path objects
            config_paths_to_absolute(experiment_configuration, Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

            # update the experiment configuration with the global keys
            for key in self.main_config:
                if key not in experiment_configuration and key != 'datasets':
                    experiment_configuration[key] = self.main_config[key]

            model_selection_evaluation(dataset['name'], Path(experiment_configuration['paths']['results']), evaluation_type=evaluation_type, evaluate_best_model=evaluate_best_model,experiment_config=experiment_configuration, evaluate_validation_only=evaluate_validation_only)

    def RunBestModel(self, dataset_names=None):
        '''
        Run over all the datasets defined in the main config file (default) or only over the datasets defined in the dataset_names list.
        '''
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        if dataset_names is not None:
            self.dataset_names = dataset_names
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            print(f"Running experiment for dataset {dataset['name']}")
            validation_folds = dataset['validation_folds']

            # load the config file
            # run the best models
            # parallelize over (run_id, validation_id) pairs
            evaluation_run_number = 3

            experiment_configuration_path = dataset.get('experiment_config_file', '')
            # load the config file
            experiment_configuration = yaml.load(open(experiment_configuration_path), Loader=yaml.FullLoader)
            paths = collect_paths(main_configuration=self.main_config, dataset_configuration=dataset,
                          experiment_configuration=experiment_configuration)
            experiment_configuration['paths'] = paths
            # paths to Path objects
            config_paths_to_absolute(experiment_configuration, Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))

            # update the experiment configuration with the global keys
            for key in self.main_config:
                if key not in experiment_configuration and key != 'datasets':
                    experiment_configuration[key] = self.main_config[key]


            parallelization_pairs = [(run_id, validation_id) for run_id in range(evaluation_run_number) for validation_id in range(validation_folds)]
            num_jobs = min(len(parallelization_pairs), os.cpu_count())
            print(f"Run the best model of dataset {dataset['name']} using {evaluation_run_number} different runs. The number of parallel jobs is {num_jobs}")
            joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(self.run_best_model)(graph_db_name=dataset['name'],
                                                                                 run_id=run_id,
                                                                                 validation_id=validation_id,
                                                                                 validation_number=validation_folds,
                                                                                 graph_format="NEL",
                                                                                 experiment_configuration=experiment_configuration,
                                                                                 evaluation_type='accuracy')
                                             for run_id, validation_id in parallelization_pairs)

    def Preprocess(self, num_jobs=-1):
        # parallelize over the datasets
        if num_jobs == -1:
            num_jobs = min(len(self.main_config['datasets']), os.cpu_count())
        joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(self.parallel_preprocessing)(dataset_configuration) for dataset_configuration in self.main_config['datasets'])

    def parallel_preprocessing(self, dataset_configuration):
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
        Preprocessing(db_name=db_name, experiment_configuration=experiment_configuration, with_splits=with_splits,
                      data_generation=data_generation, data_generation_args=data_generation_args)

    def check_config_consistency(self):
        pass

    def find_best_models(self, graph_db_name, validation_id, validation_folds, graph_format,
                         experiment_configuration, experiment_configuration_path, run_id=0):
        if experiment_configuration is not None:
            absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            absolute_path = Path(absolute_path)

            experiment_configuration['format'] = graph_format

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
            graph_data = get_graph_data(db_name=graph_db_name, data_path=data_path,
                                        use_features=experiment_configuration['use_features'],
                                        use_attributes=experiment_configuration['use_attributes'],
                                        data_format=experiment_configuration['format'])
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
                self.run_configuration(config_id=c_id, run_config=run_config, graph_data=graph_data,
                                  graph_db_name=graph_db_name, run_id=run_id, validation_id=validation_id,
                                  validation_folds=validation_folds)
            # copy config file to the results directory if it is not already there
            if not os.path.exists(r_path.joinpath(f"{graph_db_name}/config.yml")):
                source_path = Path(absolute_path).joinpath(experiment_configuration_path)
                destination_path = r_path.joinpath(f"{graph_db_name}/config.yml")
                # copy the config file to the results directory
                # if linux
                if os.name == 'posix':
                    os.system(f"cp {source_path} {destination_path}")
                # if windows
                elif os.name == 'nt':
                    os.system(f"copy {source_path} {destination_path}")
        else:
            # print that config file is not provided
            print("Please provide a configuration file")

    def run_configuration(self, config_id, run_config, graph_data: GraphData, graph_db_name, run_id, validation_id,
                          validation_folds):
        para = Parameters()
        load_preprocessed_data_and_parameters(config_id=config_id, run_id=run_id, validation_id=validation_id,
                                              validation_folds=validation_folds, graph_db_name=graph_db_name,
                                              graph_data=graph_data, run_config=run_config, para=para)
        self.validation_step(para.run_id, para.validation_id, graph_data, para)

    def validation_step(self, run_id, validation_id, graph_data: GraphData, para: Parameters):
        """
        Split the data in training validation and test set
        """
        seed = 56874687 + validation_id + para.n_val_runs * run_id
        data = Load_Splits(para.splits_path, para.db, para.run_config.config.get('transfer', False))
        test_data = np.asarray(data[0][validation_id], dtype=int)
        training_data = np.asarray(data[1][validation_id], dtype=int)
        validate_data = np.asarray(data[2][validation_id], dtype=int)

        method = ModelEvaluation(run_id, validation_id, graph_data, training_data, validate_data, test_data, seed, para)

        """
        Run the method
        """
        method.Run()

    def run_best_model(self, graph_db_name, run_id, validation_number, validation_id, graph_format, experiment_configuration, evaluation_type):
        if experiment_configuration is not None:
            experiment_configuration['format'] = graph_format
            # set best model to true
            experiment_configuration['best_model'] = True

            data_path = experiment_configuration['paths']['data']
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
            graph_data = get_graph_data(db_name=graph_db_name, data_path=data_path,
                                        use_features=experiment_configuration['use_features'],
                                        use_attributes=experiment_configuration['use_attributes'], data_format=graph_format)
            # adapt the precision of the input data
            if 'precision' in experiment_configuration:
                if experiment_configuration['precision'] == 'double':
                    for i in range(len(graph_data.inputs)):
                        graph_data.inputs[i] = graph_data.inputs[i].double()

            run_configs = get_run_configs(experiment_configuration)
            # get the best configuration and run it
            best_config_id = model_selection_evaluation(db_name=graph_db_name,
                                                        path=experiment_configuration['paths']['results'],
                                                        evaluation_type=evaluation_type, get_best_model=True,
                                                        experiment_config=experiment_configuration)
            # config_id = get_best_configuration(graph_db_name, configs, evaluation_type=evaluation_type)

            c_id = f'Best_Configuration_{str(best_config_id).zfill(6)}'
            self.run_configuration(config_id=c_id, run_config=run_configs[best_config_id], graph_data=graph_data,
                              graph_db_name=graph_db_name, run_id=run_id, validation_id=validation_id,
                              validation_folds=validation_number)
        else:
            # print that config file is not provided
            print("Please provide a configuration file")


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

