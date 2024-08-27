# create an example dataset in the correct format to run the model defined in Examples/Config/config_example.py
import networkx as nx

from Examples.CustomDataset.create_example_graphs import ring_diagonals
from src.Preprocessing.create_labels import save_standard_labels, save_wl_labels, save_cycle_labels, save_subgraph_labels
from src.Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from src.utils.utils import save_graphs
from pathlib import Path

def generate_labels(db_name:str, data_path: Path, label_path: Path):
    save_standard_labels(data_path, db_names=[db_name], label_path=label_path, format='NEL')
    save_wl_labels(data_path, db_names=[db_name], max_iterations=2, max_label_num=500, label_path=label_path, format='NEL')
    save_cycle_labels(data_path, db_names=[db_name], length_bound=100, max_node_labels=500, cycle_type='simple', label_path=label_path, format='NEL')
    save_subgraph_labels(data_path, db_names=[db_name], subgraphs=[nx.complete_graph(4)], name="subgraph", id=0, label_path=label_path, format='NEL')


def generate_properties(db_name:str, data_path: Path, property_path: Path):
    write_distance_properties(data_path, db_name=db_name, out_path=property_path, cutoff=None, data_format='NEL')
    write_distance_edge_properties(data_path, db_name=db_name, out_path=property_path, cutoff=1, data_format='NEL')


def generate_splits(db_name:str, path: Path, output_path: Path):
    generate_splits(db_name=db_name, path=path, output_path=output_path, format='NEL')

def generate_data(data_path: Path, db_name: str):
    # create example graphs
    graphs, labels = ring_diagonals(1000, 50)
    # save lists of graphs and labels in the correct format NEL -> Nodes, Edges, Labels
    save_graphs(data_path, db_name, graphs, labels, with_degree=False, format='NEL')

def main(with_data=True, with_splits=True, with_labels=True, with_properties=True):
    data_path = Path('Examples/Data/')
    split_path = data_path.joinpath('Splits/')
    label_path = data_path.joinpath('Labels/')
    property_path = data_path.joinpath('Properties/')
    db_name = 'EXAMPLE_DB'
    if with_data:
        # create example graphs
        generate_data(data_path, db_name)
    if with_splits:
        # create fixed splits for training, validation and test set
        generate_splits(db_name, data_path, split_path)
    if with_labels:
        # create labels
        generate_labels(db_name, data_path, label_path)
    if with_properties:
        generate_properties(db_name, data_path, property_path)

if __name__ == '__main__':
    main(with_data=True, with_splits=True, with_labels=True, with_properties=True)