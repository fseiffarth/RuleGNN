from pathlib import Path
from scripts.Preprocessing import Preprocessing
from scripts.run_all import run_all
from src.utils.synthetic_graphs import ring_diagonals


def main(get_data, get_data_args):
    db_name = 'EXAMPLE_DB'
    config_file =  Path('Examples/CustomDataset/Config/config_example.yml')
    # preprocess the data
    Preprocessing(db_name=db_name, config_file=config_file, with_splits=True, get_data=get_data, get_data_args=get_data_args)
    run_all(database_names=[db_name], cross_validations=[10], config_files=[config_file])
    #preprocessing(with_data=with_data, with_splits=with_splits, with_labels=with_labels, with_properties=with_properties)
    # run the experiment
    #run_all(database_names=[db_name], cross_validations=[10], config_files=[config_file])

if __name__ == '__main__':
    main(ring_diagonals, (1000, 50))