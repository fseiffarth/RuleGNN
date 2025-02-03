from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce/Data").mkdir(exist_ok=True)
    Path("Reproduce/Data/Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "CSL"]:
        source_path = Path("Data/Splits").joinpath(f"{split}_splits.json")
        target_path = Path("Reproduce/Data/Splits").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())

    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce/Data/SplitsSimple").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]:
        source_path = Path("Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path = Path("Reproduce/Data/SplitsSimple").joinpath(f"{split}_splits.json")
        target_path.write_text(source_path.read_text())



def main():
    get_existing_splits()
    for threshold in range(1,21):
        ablation_experiment = ExperimentMain(Path(f'Reproduce/Configs/ablation/threshold/upper/main_config_ablation_threshold_{threshold}.yml'))
        ablation_experiment.Preprocess()
        ablation_experiment.GridSearch()
        ablation_experiment.EvaluateResults()
        ablation_experiment.RunBestModel()
        ablation_experiment.EvaluateResults(evaluate_best_model=True)
    for threshold in range(1,21):
        ablation_experiment = ExperimentMain(Path(f'Reproduce/Configs/ablation/threshold/lower_upper/main_config_ablation_threshold_{threshold}.yml'))
        ablation_experiment.Preprocess()
        ablation_experiment.GridSearch()
        ablation_experiment.EvaluateResults()
        ablation_experiment.RunBestModel()
        ablation_experiment.EvaluateResults(evaluate_best_model=True)
    for threshold in range(1,21):
        ablation_experiment = ExperimentMain(Path(f'Reproduce/Configs/ablation/threshold/lower/main_config_ablation_threshold_{threshold}.yml'))
        ablation_experiment.Preprocess()
        ablation_experiment.GridSearch()
        ablation_experiment.EvaluateResults()
        ablation_experiment.RunBestModel()
        ablation_experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()