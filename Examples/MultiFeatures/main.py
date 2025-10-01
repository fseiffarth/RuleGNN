from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Examples/MultiFeatures/Configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    experiment.run_configurations()
    experiment.evaluate_results()
    experiment.run_best_configuration()
    experiment.evaluate_results(evaluate_best_model=True)

if __name__ == '__main__':
    main()