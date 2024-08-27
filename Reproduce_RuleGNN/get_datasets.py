# get all the datasets from the TU Dortmund benchmark graph dataset used in the paper
import os
from pathlib import Path

from src.utils.TU_to_NEL import tu_to_nel


def get_real_world_datasets():
    for db_name in ['DHFR', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'NCI1', 'NCI109']:
        tu_to_nel(db_name=db_name, out_path=Path('Data/Reproduce_RuleGNN/TUDatasets'))

# main
if __name__ == '__main__':
    # make folder TUDatasets in Reproduce_RuleGNN
    if not os.path.exists('Reproduce_RuleGNN/Data/TUDatasets'):
        os.makedirs(Path('Reproduce_RuleGNN/Data/TUDatasets'))
    get_real_world_datasets()