from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():


    ## Synthetic Data
    experiment_synthetic = ExperimentMain(Path('Reproduce/Configs/main_config_fair_synthetic.yml'))
    experiment_synthetic.Preprocess()

    ## run synthetic experiment
    experiment_synthetic.GridSearch()
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel()
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('Reproduce/Configs/main_config_fair_synthetic_random_variation.yml'))
    experiment_synthetic.Preprocess()
    experiment_synthetic.GridSearch()
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel()
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('Reproduce/Configs/main_config_fair_synthetic_only_encoder.yml'))
    experiment_synthetic.Preprocess()
    experiment_synthetic.GridSearch()
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel()
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    experiment_synthetic = ExperimentMain(Path('Reproduce/Configs/main_config_fair_synthetic_only_decoder.yml'))
    experiment_synthetic.Preprocess()
    experiment_synthetic.GridSearch()
    experiment_synthetic.EvaluateResults()
    experiment_synthetic.RunBestModel()
    experiment_synthetic.EvaluateResults(evaluate_best_model=True)


if __name__ == '__main__':
    main()