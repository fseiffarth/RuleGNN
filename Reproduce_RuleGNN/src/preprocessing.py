from pathlib import Path

import networkx as nx

from src.Preprocessing.create_labels import save_standard_labels, save_wl_labels, save_cycle_labels, \
    save_subgraph_labels, save_clique_labels
from src.Preprocessing.create_properties import write_distance_properties


def generate_labels(db_name:str, data_path: Path, label_path: Path, label_generation_times: Path):
    # save for all dataset the standard labels and WL labels
    save_standard_labels(data_path, db_names=[db_name], label_path=label_path, format='NEL', save_times=label_generation_times)
    if db_name == 'DHFR':
        save_wl_labels(data_path, db_names=[db_name], max_iterations=2, max_label_num=50000, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=10, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=20, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
    if db_name == 'CSL':
        save_cycle_labels(data_path, db_names=[db_name], length_bound=10, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
    if db_name in ['IMDB-BINARY', 'IMDB-MULTI']:
        save_cycle_labels(data_path, db_names=[db_name], length_bound=3, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=4, max_node_labels=500, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=4, max_node_labels=1000, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=5, max_node_labels=500, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=5, max_node_labels=1000, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=4, cycle_type='induced', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=5, cycle_type='induced', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=10, cycle_type='induced', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=20, cycle_type='induced', label_path=label_path, format='NEL', save_times=label_generation_times)

        # clique labels
        save_clique_labels(data_path, db_names=[db_name], max_clique=3, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_clique_labels(data_path, db_names=[db_name], max_clique=4, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_clique_labels(data_path, db_names=[db_name], max_clique=6, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_clique_labels(data_path, db_names=[db_name], max_clique=10, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_clique_labels(data_path, db_names=[db_name], max_clique=20, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_clique_labels(data_path, db_names=[db_name], max_clique=50, label_path=label_path, format='NEL', save_times=label_generation_times)

        # wl labels
        save_wl_labels(data_path, db_names=[db_name], max_iterations=1, max_label_num=50000, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_wl_labels(data_path, db_names=[db_name], max_iterations=2, max_label_num=50000, label_path=label_path, format='NEL', save_times=label_generation_times)

        # subgraph labels
        save_subgraph_labels(data_path, db_names=[db_name], subgraphs=[nx.complete_graph(4)], id=0, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_subgraph_labels(data_path, db_names=[db_name], subgraphs=[nx.cycle_graph(3), nx.star_graph(1)], id=1, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_subgraph_labels(data_path, db_names=[db_name], subgraphs=[nx.cycle_graph(4), nx.star_graph(1)], id=2, label_path=label_path, format='NEL', save_times=label_generation_times)
        save_subgraph_labels(data_path, db_names=[db_name], subgraphs=[nx.cycle_graph(3), nx.cycle_graph(4), nx.star_graph(1)], id=3, label_path=label_path, format='NEL', save_times=label_generation_times)
    if db_name in ['NCI1', 'NCI109', 'Mutagenicity']:
        save_cycle_labels(data_path, db_names=[db_name], length_bound=6, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=8, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)
        save_cycle_labels(data_path, db_names=[db_name], length_bound=10, cycle_type='simple', label_path=label_path, format='NEL', save_times=label_generation_times)

        # wl
        save_wl_labels(data_path, db_names=[db_name], max_iterations=2, max_label_num=50000, label_path=label_path, format='NEL', save_times=label_generation_times)

def generate_properties(db_name:str, data_path: Path, property_path: Path, property_generation_times: Path):
    write_distance_properties(data_path, db_name=db_name, out_path=property_path, cutoff=None, data_format='NEL', save_times=property_generation_times)

def preprocessing(with_labels=True, with_properties=True):
    # preprocesss real-world datasets
    tu_data_path = Path('Reproduce_RuleGNN/Data/TUDatasets/')
    synthetic_data_path = Path('Reproduce_RuleGNN/Data/SyntheticDatasets/')
    label_path = Path('Reproduce_RuleGNN/Data/Labels/')
    # make file to store generation times
    with open('Reproduce_RuleGNN/Data/generation_times_labels.txt', 'w') as f:
        f.write('Database, LabelName, Time\n')
    label_generation_times = Path('Reproduce_RuleGNN/Data/generation_times_labels.txt')
    property_path = Path('Reproduce_RuleGNN/Data/Properties/')
    # make file to store generation times
    with open('Reproduce_RuleGNN/Data/generation_times_properties.txt', 'w') as f:
        f.write('Database, PropertyName, Time\n')
    property_generation_times = Path('Reproduce_RuleGNN/Data/generation_times_properties.txt')
    real_world_db_names = ['NCI1', 'NCI109', 'Mutagenicity', 'DHFR', 'IMDB-BINARY', 'IMDB-MULTI']
    synthetic_db_names = ['Snowflakes', 'CSL', 'EvenOddRings2_16', 'EvenOddRingsCount16', 'LongRings100']
    for db_name in real_world_db_names:
        if with_labels:
            # create labels
            generate_labels(db_name, tu_data_path, label_path, label_generation_times)
        if with_properties:
            generate_properties(db_name, tu_data_path, property_path, property_generation_times)
    for db_name in synthetic_db_names:
        if with_labels:
            # create labels
            generate_labels(db_name, synthetic_data_path, label_path, label_generation_times)
        if with_labels:
            generate_properties(db_name, synthetic_data_path, property_path, property_generation_times)



if __name__ == '__main__':
    preprocessing(with_labels=True, with_properties=True)