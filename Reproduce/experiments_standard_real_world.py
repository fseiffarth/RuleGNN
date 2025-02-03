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

    experiment = ExperimentMain(Path('Reproduce/Configs/main_config_sota_comparison.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults(evaluate_validation_only=True)

    experiment = ExperimentMain(Path('Reproduce/Configs/main_config_sota_random_comparison.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults(evaluate_validation_only=True)


if __name__ == '__main__':
    main()
