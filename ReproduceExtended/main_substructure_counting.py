## Real World Data
from pathlib import Path

import click

from Reproduce.latex_plots import plot_specific_graphs_from_db
from scripts.Evaluation.graph_plotting import plot_all_graphs_from_db
from src.Experiment.ExperimentMain import ExperimentMain

def main_counting(num_threads=-1):

    experiment = ExperimentMain(Path('ReproduceExtended/configs/main_config_substructure_counting.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    # plotting the graphs
    #plot_all_graphs_from_db( 'cycle6', experiment)
    ## run real world experiment
    experiment.GridSearch(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_counting(num_threads)



if __name__ == '__main__':
    main()