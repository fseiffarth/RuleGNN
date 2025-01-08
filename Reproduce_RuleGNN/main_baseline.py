from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.ExperimentMain import ExperimentMain
from src.Competitors.Kernels.GraphKernels import WLKernel
from src.Competitors.Kernels.NoGKernel import NoGKernel
from src.utils.GraphData import get_graph_data
from src.utils.load_splits import Load_Splits


def validation(dataset, experiment_configuration, validation_id, graph_data):
    # three runs
    splits = Load_Splits(experiment_configuration['paths']['splits'], dataset)
    test_data = np.asarray(splits[0][validation_id], dtype=int)
    training_data = np.asarray(splits[1][validation_id], dtype=int)
    validate_data = np.asarray(splits[2][validation_id], dtype=int)
    out_path = experiment_configuration['paths']['results']
    # make dir competitors under the out_path
    out_path = Path(out_path).joinpath("Baseline").joinpath(dataset)
    out_path.mkdir(exist_ok=True, parents=True)


    noG = NoGKernel(out_path=out_path, graph_data=graph_data, run_num=0, validation_num=validation_id, training_data=training_data,
                    validate_data=validate_data, test_data=test_data, seed=42)
    noG.Run()

    wlKernel = WLKernel(out_path=out_path, graph_data=graph_data, run_num=0, validation_num=validation_id,
                        training_data=training_data, validate_data=validate_data, test_data=test_data,
                        seed=42)
    wlKernel.Run()




def evaluation(dataset, experiment_configuration, graph_data, algorithm, test=True):
    '''
    Evaluate the results of the competitors
    '''
    evaluation = {}

    # load the data from Results/{db_name}/Results/{db_name}_{id_str}_Results_run_id_{run_id}.csv as a pandas dataframe for all run_ids in the directory
    # ge all those files
    files = []
    for file in experiment_configuration['paths']['results'].joinpath('Baseline').joinpath(dataset).iterdir():
        files.append(file)

    df_all = None
    for i, file in enumerate(files):
        df = pd.read_csv(file, delimiter=";")
        # concatenate the dataframes
        if df_all is None:
            df_all = df
        else:
            df_all = pd.concat([df_all, df], ignore_index=True)

    # get all rows where the algorithm is NoGKernel
    df_all = df_all[df_all['Algorithm'] == algorithm]
    # if the column HyperparameterAlgo is not present, add it with the value 0
    if 'HyperparameterAlgo' not in df_all.columns:
        df_all['HyperparameterAlgo'] = 0

    # group by the hyperparametersvc and the hyperparameterAlgo
    groups = df_all.groupby(['HyperparameterSVC', 'HyperparameterAlgo'])
    # groups = df_all.groupby('HyperparameterSVC')

    evaluation = []
    for name, group in groups:
        df_validation = group.groupby('ValidationNumber').mean(numeric_only=True)
        # get the average and deviation over all runs
        avg = df_validation.mean(numeric_only=True)
        std = df_validation.std()
        evaluation.append(
            {'HyperparameterSVC': avg['HyperparameterSVC'], 'HyperparameterAlgo': avg['HyperparameterAlgo'],
             'ValidationAccuracy': round(100 * avg['ValidationAccuracy'], 2),
             'ValidationAccuracyStd': round(100 * std['ValidationAccuracy'], 2),
             'TestAccuracy': round(100 * avg['TestAccuracy'], 2),
             'TestAccuracyStd': round(100 * std['TestAccuracy'], 2)})
        # print the avg and std together with the hyperparameter and the algorithm used
        #print(f"Hyperparameter: {name} Average Test Accuracy: {avg['TestAccuracy']} +/- {std['TestAccuracy']}")

    # get the three best hyperparameters according to the average validation accuracy
    evaluation = sorted(evaluation, key=lambda x: x['ValidationAccuracy'], reverse=True)
    best_hyperparameters = evaluation[:3]
    # print the three best hyperparameters
    print(f"{dataset} Best hyperparameters:")
    for hyperparameter in best_hyperparameters:
        print(
            f"Hyperparameter: SVC:{hyperparameter['HyperparameterSVC']} Algo:{hyperparameter['HyperparameterAlgo']} "
            f"Validation Accuracy: {hyperparameter['ValidationAccuracy']} +/- {hyperparameter['ValidationAccuracyStd']} "
            f"Test Accuracy: {hyperparameter['TestAccuracy']} +/- {hyperparameter['TestAccuracyStd']}")


def main_baseline():
    '''
    Run the baseline models for the given dataset
    '''
    # load the yml file
    for config in ['main_config_fair_real_world.yml', 'main_config_sota_comparison.yml', 'main_config_fair_synthetic.yml']:
        experiment = ExperimentMain(Path(f"Reproduce_RuleGNN/Configs/{config}"))
        datasets = list(experiment.experiment_configurations.keys())
        for dataset in datasets:
            experiment_configuration = experiment.experiment_configurations[dataset]
            # load the graph data
            graph_data = get_graph_data(db_name=dataset, data_path=experiment_configuration['paths']['data'],
                                        task=experiment_configuration.get('task', 'graph_classification'),
                                        input_features=experiment_configuration.get('input_features', None),
                                        output_features=experiment_configuration.get('output_features', None),
                                        graph_format=experiment_configuration.get('format', 'RuleGNNDataset'),
                                        precision=experiment_configuration.get('precision', 'double'))

            validation_size = 10
            if dataset == "CSL":
                validation_size = 5
            # run the validation for all validation sets in parallel
            joblib.Parallel(n_jobs=validation_size)(
                joblib.delayed(validation)(dataset, experiment_configuration, validation_id, graph_data) for validation_id in range(validation_size))


            test = True
            if config == 'main_config_sota_comparison.yml':
                test = False
            evaluation(dataset, experiment_configuration, graph_data, algorithm='NoGKernel', test=test)
            evaluation(dataset, experiment_configuration, graph_data, algorithm='WLKernel', test=test)

if __name__ == "__main__":
    main_baseline()
