from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/TextClassification/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.run_configurations()
    experiment.evaluate_results()


if __name__ == '__main__':
    main()