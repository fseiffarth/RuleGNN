from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def get_existing_splits():
    # copy the splits from the Data folder to the Splits folder
    # create the Splits folder if it does not exist
    Path("Reproduce_RuleGNN/Data").mkdir(exist_ok=True)
    Path("Reproduce_RuleGNN/Data/Splits").mkdir(exist_ok=True)
    # copy the splits for NCI1, IMDB-BINARY, IMDB-MULTI and CSL
    for split in ["NCI1", "IMDB-BINARY", "IMDB-MULTI", "CSL"]:
        Path.write_text(Path("Reproduce_RuleGNN/Data/Splits").joinpath(f"{split}_splits.json"), Path("Data/Splits").joinpath(f"{split}_splits.json").read_text())



def main():
    get_existing_splits()

    ### Synthetic Data
    #experiment_synthetic = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_synthetic.yml'))
    #experiment_synthetic.Preprocess()
    #experiment_synthetic.GridSearch()
    #experiment_synthetic.EvaluateResults()
    #experiment_synthetic.RunBestModel()
    #experiment_synthetic.EvaluateResults(evaluate_best_model=True)

    ### Real World Data
    experiment = ExperimentMain(Path('Reproduce_RuleGNN/Configs/main_config_fair_real_world.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()