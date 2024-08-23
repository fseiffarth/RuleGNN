# create an example dataset in the correct format to run the model defined in Example/Config/config_example.py
import networkx as nx

from Example.create_example_graphs import ring_diagonals
from Preprocessing.create_labels import save_standard_labels, save_wl_labels, save_circle_labels, save_subgraph_labels
from Preprocessing.create_properties import write_distance_properties, write_distance_edge_properties
from Preprocessing.create_splits import create_splits
from utils.utils import save_graphs

def create_labels(data_path):
    label_path = data_path + 'Labels/'
    save_standard_labels(data_path, db_names=['EXAMPLE_DB'], label_path=label_path, format='NEL')
    save_wl_labels(data_path, db_names=['EXAMPLE_DB'], max_iterations=2, max_label_num=500, label_path=label_path, format='NEL')
    save_circle_labels(data_path, db_names=['EXAMPLE_DB'], length_bound=100, max_node_labels=500, cycle_type='simple', label_path=label_path, format='NEL')
    save_subgraph_labels(data_path, db_names=['EXAMPLE_DB'], subgraphs=[nx.complete_graph(4)], name="subgraph", id=0, label_path=label_path, format='NEL')


def create_properties(data_path):
    write_distance_properties(data_path, db_name='EXAMPLE_DB', out_path=data_path + 'Properties/', cutoff=None, data_format='NEL')
    write_distance_edge_properties(data_path, db_name='EXAMPLE_DB', out_path=data_path + 'Properties/', cutoff=1, data_format='NEL')



def main(save_data=True, with_splits=True, with_labels=True, with_properties=True):
    data_path = 'Example/Data/'
    if save_data:
        # create example graphs
        graphs, labels = ring_diagonals(1000, 50)
        # save lists of graphs and labels in the correct format NEL -> Nodes, Edges, Labels
        save_graphs(data_path, 'EXAMPLE_DB', graphs, labels, with_degree=False, format='NEL')
    if with_splits:
        # create fixed splits for training, validation and test set
        create_splits(db_name='EXAMPLE_DB', path=data_path, output_path=data_path + 'Splits/', format='NEL')
    if with_labels:
        # create labels
        create_labels(data_path)
    if with_properties:
        create_properties(data_path)

if __name__ == '__main__':
    main(save_data=True, with_splits=True, with_labels=True, with_properties=True)