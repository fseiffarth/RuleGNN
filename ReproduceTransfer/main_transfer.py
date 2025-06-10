## Real World Data
import json
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain
from src.TransferLearning.combine_split_files import pretraining_finetuning
from src.utils.load_splits import Load_Splits


def preprocessing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    target_path = Path("ReproduceTransfer/Data/SplitsTransfer")
    target_path.mkdir(exist_ok=True, parents=True)

    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "Mutagenicity"]:
        source_file = Path("Data/SplitsTransfer").joinpath(f"{split}_splits.json")
        target_file = target_path.joinpath(f"{split}_splits.json")
        if not target_file.exists():
            target_file.write_text(source_file.read_text())
    datasets = ['NCI1', 'NCI109', 'Mutagenicity']
    paths = [Path("ReproduceTransfer/Data/SplitsTransfer") for split in datasets]
    pretraining_finetuning(paths, datasets, pretraining_ids=[0,1], finetuning_ids=[2])
    pass



def transfer(num_threads=-1):
    # load the model and fine-tune it on the target dataset
    experiment_pretrained = ExperimentMain(Path('ReproduceTransfer/configs_transfer/main_config_pretraining.yml'))
    experiment_pretrained.ExperimentPreprocessing(num_threads=num_threads)
    for i, experiment_configuration in enumerate(experiment_pretrained.experiment_configurations['NCI1_NCI109_Mutagenicity']):
        # get pretraining_datasets from the experiment configuration
        pretraining_datasets = experiment_configuration.get('pretraining_datasets', None)
        if pretraining_datasets is None:
            raise ValueError("pretraining_datasets must be specified in the experiment configuration")
        # finetuning
        experiment_finetuning = ExperimentMain(Path(f'ReproduceTransfer/configs_transfer/main_config_finetune.yml'), pretrained_network=(experiment_pretrained, i))
        # set results appendix
        for config in experiment_finetuning.experiment_configurations['NCI1_NCI109_Mutagenicity']:
            config['results_appendix'] = f"{experiment_configuration['results_appendix']}_" + config["results_appendix"]
            # also set the results path
            config['paths']['results'] = config['paths']['results'].joinpath(experiment_configuration['results_appendix'])
        experiment_finetuning.ExperimentPreprocessing(num_threads=num_threads)
        ## run real world experiment using the pre-trained model
        experiment_finetuning.GridSearch(num_threads=num_threads)
        experiment_finetuning.EvaluateResults()
        experiment_finetuning.RunBestModel(num_threads=num_threads)
        experiment_finetuning.EvaluateResults(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    preprocessing_splits()
    experiment = ExperimentMain(Path('ReproduceTransfer/configs_transfer/main_config_pretraining.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    ## run real world experiment
    experiment.GridSearch(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)



if __name__ == '__main__':
    #main()
    transfer()