import os
from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():
    # export omp_num_threads=1
    os.environ["OMP_NUM_THREADS"] = "1"
    experiment = ExperimentMain(Path('Examples/PyG/Configs/config_main.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()