## Real World Data
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain

def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce/Data").mkdir(exist_ok=True)
    Path("Reproduce/Data/Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "CSL"]:
        source_path = Path("Data/Splits").joinpath(f"{split}_splits.json")
        target_path = Path("ReproduceExtended/Data/Splits").joinpath(f"{split}_splits.json")
        if not target_path.exists():
            target_path.write_text(source_path.read_text())


def transfer(num_threads=-1):
    # load the model and fine-tune it on the target dataset
    experiment = ExperimentMain(Path('ReproduceExtended/configs_transfer/main_config.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)



    run_id = 0
    validation_id = 0
    datasets = ['NCI1', 'NCI109', 'Mutagenicity']
    for db_name in datasets:
        # fine-tune the model on the target dataset, i.e., all except for db_name
        for target_db_name in datasets:
            if target_db_name != db_name:
                print(f'Fine-tuning {target_db_name} on model pre-trained on {db_name}')
                model = experiment.load_model(db_name, run_id=run_id, validation_id=validation_id)
                experiment_finetune = ExperimentMain(Path(f'ReproduceExtended/configs_transfer/main_config_finetune_{target_db_name}.yml'),
                                                     pretrained_network=model)
                experiment_finetune.ExperimentPreprocessing(num_threads=num_threads)
                ## run real world experiment
                experiment_finetune.GridSearch(num_threads=num_threads)
                experiment_finetune.EvaluateResults()
                experiment_finetune.RunBestModel(num_threads=num_threads)
                experiment_finetune.EvaluateResults(evaluate_best_model=True)


@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    get_existing_splits()
    experiment = ExperimentMain(Path('ReproduceExtended/configs_transfer/main_config.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    ## run real world experiment
    experiment.GridSearch(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)



if __name__ == '__main__':
    #main()
    transfer()