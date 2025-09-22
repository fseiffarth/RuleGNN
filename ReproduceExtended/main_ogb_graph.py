## Real World Data
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain


def main_ogb_graph(num_threads=-1):
    experiment = ExperimentMain(Path('ReproduceExtended/configs/OGB_Graph/main_config_OGB.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    ## run real world experiment
    experiment.GridSearch(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_ogb_graph(num_threads)



if __name__ == '__main__':
    main()