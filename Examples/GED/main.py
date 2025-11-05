## Real World Data
from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain

def main_example_gcn(num_threads=-1):
    # Load and preprocess the experiment
    experiment = ExperimentMain(Path('Examples/GED/Configs/main_config.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)

    # Run and evaluate all configurations defined in the config file
    experiment.run_configurations(num_threads=num_threads)
    experiment.evaluate_results()
    # Run and evaluate the best configuration
    experiment.run_best_configuration(num_threads=num_threads)
    experiment.evaluate_results(evaluate_best_model=True)

@click.command()
@click.option('--num_threads', default=1, help='Number of threads to use')
def main(num_threads):
    main_example_gcn(num_threads)



if __name__ == '__main__':
    main()