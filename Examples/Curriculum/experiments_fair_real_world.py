from pathlib import Path

import click

from src.Experiment.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce_RuleGNN/Data").mkdir(exist_ok=True)
    Path("Reproduce_RuleGNN/Data/Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "CSL"]:
        source_path = Path("Data/Splits").joinpath(f"{split}_splits.json")
        target_path = Path("Reproduce_RuleGNN/Data/Splits").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())

    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce_RuleGNN/Data/SplitsSimple").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]:
        source_path = Path("Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path = Path("Reproduce_RuleGNN/Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())


def main_fair_real_world(num_threads=-1):
    get_existing_splits()

    ## Real World Data
    experiment = ExperimentMain(Path('Experiments/Curriculum/configs/main_config_fair_real_world_curriculum.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    experiment.run_configurations(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

    experiment = ExperimentMain(Path('Experiments/Curriculum/configs/main_config_fair_real_world_anti_curriculum.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    experiment.run_configurations(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

    experiment = ExperimentMain(Path('Experiments/Curriculum/configs/main_config_fair_real_world_curriculum_edges.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    experiment.run_configurations(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

    experiment = ExperimentMain(Path('Experiments/Curriculum/configs/main_config_fair_real_world_anti_curriculum_edges.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    experiment.run_configurations(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)

    experiment = ExperimentMain(Path('Experiments/Curriculum/configs/main_config_fair_real_world.yml'))
    experiment.ExperimentPreprocessing(num_threads=num_threads)
    experiment.run_configurations(num_threads=num_threads)
    experiment.EvaluateResults()
    experiment.RunBestModel(num_threads=num_threads)
    experiment.EvaluateResults(evaluate_best_model=True)



@click.command()
@click.option('--num_threads', default=-1, help='Number of threads to use')
def main(num_threads):
    main_fair_real_world(num_threads=num_threads)

if __name__ == '__main__':
    main()