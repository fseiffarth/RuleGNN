import os
from pathlib import Path

import yaml

from ExperimentMain import collect_paths
from Preprocessing import Preprocessing
from scripts.ExperimentMain import ExperimentMain
from src.utils.combine_nel import combine_nel_graphs
from src.utils.path_conversions import config_paths_to_absolute


def main():
    # example on how to combine multiple datasets: TODO move this to main config file (key: name should be a list)
    main_configuration = yaml.safe_load(open('Examples/Transfer/Configs/config_main.yml'))
    experiment_configuration = yaml.safe_load(open('Examples/Transfer/Configs/config_experiment.yml'))
    if 'paths' not in experiment_configuration:
        experiment_configuration['paths'] = {}
    paths = collect_paths(main_configuration=main_configuration, experiment_configuration=experiment_configuration, dataset_configuration=None)
    experiment_configuration['paths'] = paths
    config_paths_to_absolute(experiment_configuration, absolute_path=Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))))


    Preprocessing('PTC_MR', experiment_configuration=experiment_configuration, data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
    Preprocessing('PTC_FM', experiment_configuration=experiment_configuration, data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
    combine_nel_graphs(dataset_names=['PTC_MR', 'PTC_FM'], input_dir=Path('Examples/Transfer/Data/'), output_dir=Path('Examples/Transfer/Data/'))
    ###
    experiment = ExperimentMain(Path('Examples/Transfer/Configs/config_main.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()