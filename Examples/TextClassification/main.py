from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/TextClassification/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.GridSearch()
    experiment.EvaluateResults()


if __name__ == '__main__':
    main()