from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/HyperparameterOptimization/configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.HyperparameterOptimization(num_threads=1)
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()