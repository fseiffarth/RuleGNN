from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/CustomExample/Configs/config_main.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.RunBestModel()
    experiment.EvaluateResults()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()