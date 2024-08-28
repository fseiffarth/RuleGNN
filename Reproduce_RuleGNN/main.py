import os

import joblib

from Reproduce_RuleGNN.src.get_datasets import get_real_world_datasets
from Reproduce_RuleGNN.src.preprocessing import preprocessing
from scripts.Preprocessing import preprocessing_from_config
from scripts.run_all import run_all


def main(with_data_generation=True, with_preprocessing=True, with_experiment=True):
    if with_data_generation:
        get_real_world_datasets()
    if with_preprocessing:
        preprocessing(with_labels=True, with_properties=True)
    if with_experiment:
        # use experiments.sh
        database_names = ['CSL', 'EvenOddRingsCount16', 'LongRings100', 'Snowflakes', 'NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
        cross_validations = [5, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        config_files = ['Reproduce_RuleGNN/Configs/config_CSL.yml', 'Reproduce_RuleGNN/Configs/config_EvenOddRings.yml', 'Reproduce_RuleGNN/Configs/config_EvenOddRingsCount.yml', 'Reproduce_RuleGNN/Configs/config_LongRings.yml', 'Reproduce_RuleGNN/Configs/config_Snowflakes.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_NCI1.yml', 'Reproduce_RuleGNN/Configs/config_DHFR.yml', 'Reproduce_RuleGNN/Configs/config_IMDB.yml', 'Reproduce_RuleGNN/Configs/config_IMDB.yml']
        # run the experiment
        run_all(database_names, cross_validations, config_files)


if __name__ == '__main__':
    main(with_data_generation=True, with_preprocessing=True, with_experiment=True)