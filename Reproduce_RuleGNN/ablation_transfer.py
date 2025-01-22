from pathlib import Path

from scripts.ExperimentMain import ExperimentMain
from src.Preprocessing.create_splits import create_transfer_splits
from src.TransferLearning.combine_graphs import combine_nel_graphs


def main():


    #Preprocessing('PTC_MR', experiment_configuration=experiment_configuration, data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
    #Preprocessing('PTC_FM', experiment_configuration=experiment_configuration, data_generation='TUDataset', with_splits=False, with_labels_and_properties=False)
    combine_nel_graphs(data_path=Path('Reproduce_RuleGNN/DataGNNComparison/'),
                       output_path=Path('Reproduce_RuleGNN/Data/Transfer/'),
                       dataset_names=['IMDB-BINARY', 'IMDB-MULTI'])
    ####################
    create_transfer_splits(db_name='IMDB-BINARY_IMDB-MULTI',
                           path=Path('Reproduce_RuleGNN/Data/Transfer/'),
                           output_path=Path('Reproduce_RuleGNN/Data/Transfer/Splits/'),
                           data_format='NEL',
                           split_type='transfer')
    create_transfer_splits(db_name='IMDB-BINARY_IMDB-MULTI',
                           path=Path('Reproduce_RuleGNN/Data/Transfer/'),
                           output_path=Path('Reproduce_RuleGNN/Data/Transfer/Splits/'),
                           data_format='NEL',
                           split_type='mixed')

    experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/ablation/transfer/main_config_ablation_transfer.yml'))
    experiment.Preprocess(num_jobs=1)
    experiment.GridSearch()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

    net = experiment.load_model('PTC_MR_PTC_FM', 0, 0, 0)

    experiment_finetune = ExperimentMain(Path('Examples/Transfer/Configs/config_finetune.yml'))
    experiment_finetune.Preprocess()
    experiment_finetune.GridSearch()
    experiment_finetune.EvaluateResults()
    experiment_finetune.RunBestModel()
    experiment_finetune.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()
