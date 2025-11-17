## Real World Data
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain

def train_ged(num_threads=-1):
    datasets = ['NCI1', 'DHFR', 'NCI109', 'Mutagenicity', 'MUTAG']

    # Load and preprocess the experiment
    for dataset in datasets:
        experiment = ExperimentMain(Path(f'Examples/GED/Configs/main_config_{dataset}.yml'))
        experiment.ExperimentPreprocessing(num_threads=num_threads)

        # Run and evaluate all configurations defined in the config file
        experiment.run_configurations(num_threads=num_threads)
        experiment.evaluate_results(evaluate_validation_only=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    train_ged(num_threads)



if __name__ == '__main__':
    main()