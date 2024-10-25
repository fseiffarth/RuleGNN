from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce_RuleGNN/Data").mkdir(exist_ok=True)
    Path("Reproduce_RuleGNN/Data/SplitsSimple").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "NCI109", "IMDB-BINARY", "IMDB-MULTI"]:
        Path.write_text(Path("Reproduce_RuleGNN/Data/SplitsSimple").joinpath(f"{split}_splits.json"), Path("Data/SplitsSimple").joinpath(f"{split}_splits.json").read_text())



def main():
    get_existing_splits()
    experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_sota_comparison.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults(evaluate_validation_only=True)

if __name__ == '__main__':
    main()