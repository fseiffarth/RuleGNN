from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.run_configurations()
    experiment.run_best_configuration()

if __name__ == '__main__':
    main()