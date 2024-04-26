import pickle

import networkx as nx
from torch_geometric.datasets import ZINC

import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from GraphData import GraphData
from GraphData.GraphData import zinc_to_graph_data
from LoadData.csl import CSL


def write_distances(graph_data, db_name, cutoff, distance_path="") -> int:
    max_distance = 0
    distances = []
    for graph in graph_data.graphs:
        d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
        # order the dictionary by the values
        for _, value in d.items():
            for _, distance in value.items():
                if distance > max_distance:
                    max_distance = distance
        distances.append(d)
    # save list of dictionaries to a pickle file
    pickle.dump(distances, open(f"{distance_path}{db_name}_distances.pkl", 'wb'))
    return max_distance


def save_distances(data_path="../../../GraphData/DS_all/", db_names=[], cutoff=None, distance_path=""):
    max_distance = 0
    for db_name in db_names:
        # load the graph data
        if db_name == 'CSL':
            csl = CSL()
            graph_data = csl.get_graphs(with_distances=False)
            max_distance = write_distances(graph_data, db_name, cutoff)
        elif db_name == 'ZINC':
            zinc_train = ZINC(root="../ZINC/", subset=True, split='train')
            zinc_val = ZINC(root="../ZINC/", subset=True, split='val')
            zinc_test = ZINC(root="../ZINC/", subset=True, split='test')
            graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC")
            max_distance = write_distances(graph_data, db_name, cutoff)
        elif db_name == 'LongRings':
            graph_data = GraphData.GraphData()
            graph_data.load_from_benchmark("LongRings", "Data/")
            max_distance = write_distances(graph_data, db_name, cutoff, distance_path)
        else:
            graph_data = gdtgl.graph_data_to_graph_list(data_path, db_name, relabel_nodes=False)
            graph_data = graph_data[0]
            max_distance = write_distances(graph_data, db_name, cutoff, distance_path)


def main():
    #save_distances(db_names=['NCI1', 'NCI109', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'DHFR', 'SYNTHETICnew'])
    #save_distances(db_names=[['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']], cutoff=2)
    save_distances(db_names=['LongRings'], cutoff=None)


if __name__ == '__main__':
    main()
