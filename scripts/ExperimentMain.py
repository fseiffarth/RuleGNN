import os
from pathlib import Path

import joblib
import yaml

from scripts.Evaluation.EvaluationFinal import model_selection_evaluation, best_model_evaluation
from scripts.Preprocessing import Preprocessing
import src.utils.SyntheticGraphs as synthetic_graphs
from scripts.find_best_models import find_best_models
from scripts.run_best_models import run_best_models


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
            config_file = dataset.get('experiment_config_file', '')
            num_runs = dataset.get('num_runs', 1)
            # load the config file
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)

            for run_id in range(num_runs):
                print(f"Run {run_id + 1} of {num_runs}")
                # find the best model hyperparameters using grid search and cross-validation
                print(f"Find the best hyperparameters for dataset {dataset['name']} using {validation_folds}-fold cross-validation and {validation_folds} number of parallel jobs")
                joblib.Parallel(n_jobs=validation_folds)(
                    joblib.delayed(find_best_models)(graph_db_name=dataset['name'],
                                                     validation_folds=validation_folds,
                                                     validation_id=validation_id, graph_format="NEL", transfer=None,
                                                     config=config_file, run_id=run_id) for validation_id in range(validation_folds))

    def EvaluateResults(self, dataset_names=None, evaluation_type='accuracy', evaluate_best_model=False):
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
            config_file = dataset.get('experiment_config_file', '')
            # load the config file
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            model_selection_evaluation(dataset['name'], Path(config['paths']['results']), evaluation_type=evaluation_type, evaluate_best_model=evaluate_best_model)

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
            config_file = dataset['experiment_config_file']
            # load the config file
            # run the best models
            # parallelize over (run_id, validation_id) pairs
            evaluation_run_number = 3
            print(f"Run the best model of dataset {dataset['name']} using {evaluation_run_number} different runs. The number of parallel jobs is {evaluation_run_number * validation_folds}")
            parallelization_pairs = [(run_id, validation_id) for run_id in range(evaluation_run_number) for validation_id in range(validation_folds)]
            num_jobs = len(parallelization_pairs)
            joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(run_best_models)(graph_db_name=dataset['name'], run_id=run_id,
                                                                             validation_id=validation_id,
                                                                             validation_number=validation_folds,
                                                                             graph_format="NEL", transfer=None,
                                                                             config=config_file,
                                                                             evaluation_type='accuracy')
                                             for run_id, validation_id in parallelization_pairs)




    def EvaluateBestModel(self, dataset_names=None, evaluation_type='accuracy'):
        # set omp_num_threads to 1 to avoid conflicts with OpenMP
        os.environ['OMP_NUM_THREADS'] = '1'
        if dataset_names is not None:
            self.dataset_names = dataset_names
        # iterate over the databases
        for dataset in self.main_config['datasets']:
            print(f"Running experiment for dataset {dataset['name']}")
            validation_folds = dataset.get('validation_folds', 10)
            config_file = dataset.get('experiment_config_file', '')
            num_runs = dataset.get('num_runs', 1)
            # load the config file
            config = yaml.load(open(config_file), Loader=yaml.FullLoader)
            best_model_evaluation(dataset['name'], Path(config['paths']['results']), evaluation_type=evaluation_type)


    def Preprocess(self, num_jobs=-1):
        # parallelize over the datasets
        if num_jobs == -1:
            num_jobs = min(len(self.main_config['datasets']), os.cpu_count())
        joblib.Parallel(n_jobs=num_jobs)(joblib.delayed(self.parallel_preprocessing)(experiment_configuration) for experiment_configuration in self.main_config['datasets'])

    def parallel_preprocessing(self, experiment_configuration):
        db_name = experiment_configuration['name']
        experiment_config_file = experiment_configuration['experiment_config_file']
        data_generation = None
        data_generation_args = None
        if 'type' in experiment_configuration:
            if experiment_configuration['type'] == 'generate_from_function':
                if 'generate_function' in experiment_configuration:
                    data_generation = getattr(synthetic_graphs, experiment_configuration['generate_function'])
                    if 'generate_function_args' in experiment_configuration:
                        data_generation_args = experiment_configuration['generate_function_args']
                    else:
                        data_generation_args = None
                else:
                    data_generation = None
                    data_generation_args = None
            elif experiment_configuration['type'] == 'TUDataset':
                data_generation = 'TUDataset'
                data_generation_args = None
            else:
                print(
                    f"The type {experiment_configuration['type']} is not supported. Please use 'generate_from_function' or 'TUDataset'")

        if 'with_splits' in experiment_configuration:
            with_splits = experiment_configuration['with_splits']
        else:
            with_splits = True

        # preprocess the data
        Preprocessing(db_name=db_name, config_file=experiment_config_file, with_splits=with_splits,
                      data_generation=data_generation, data_generation_args=data_generation_args)

    def check_config_consistency(self):
        pass

