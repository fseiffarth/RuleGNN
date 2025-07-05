## Real World Data
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain

def main_ZINC(num_threads=-1):
    for i in [3,4]:
        experiment = ExperimentMain(Path(f'ReproduceExtended/configs/ZINC/main_config_ZINC_{i}.yml'))
        experiment.ExperimentPreprocessing(num_threads=num_threads)
        ## run real world experiment
        experiment.GridSearch(num_threads=num_threads)
        experiment.EvaluateResults()
        experiment.RunBestModel(num_threads=num_threads)
        experiment.EvaluateResults(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_ZINC(num_threads)



if __name__ == '__main__':
    main()