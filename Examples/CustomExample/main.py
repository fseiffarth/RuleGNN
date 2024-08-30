from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/CustomExample/Configs/main_config.yml'))
    experiment.Preprocess()
    experiment.Run()

if __name__ == '__main__':
    main()