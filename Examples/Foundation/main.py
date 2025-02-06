## Real World Data
from pathlib import Path

import click

from scripts.ExperimentMain import ExperimentMain

def main_foundation(num_threads=-1):
    experiment = ExperimentMain(Path('Examples/Foundation/Configs/main_config.yml'))
    experiment.Preprocess(num_threads=num_threads)
    ## run real world experiment
    experiment.GridSearch(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_foundation(num_threads)



if __name__ == '__main__':
    main()