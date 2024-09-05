from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/TUExample/Configs/config_main.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.RunBestModel()

if __name__ == '__main__':
    main()