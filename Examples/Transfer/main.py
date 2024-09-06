from pathlib import Path

from Preprocessing import Preprocessing
from scripts.ExperimentMain import ExperimentMain
from src.utils.combine_nel import combine_nel_graphs


def main():
    # example on how to combine multiple datasets: TODO move this to main config file (key: name should be a list)
    Preprocessing('PTC_MR', Path('Examples/Transfer/Configs/config_experiment.yml'), data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
    Preprocessing('PTC_FM', Path('Examples/Transfer/Configs/config_experiment.yml'), data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
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