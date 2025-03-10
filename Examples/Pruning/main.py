from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.GridSearch()
    experiment.RunBestModel()

if __name__ == '__main__':
    main()