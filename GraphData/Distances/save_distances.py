import collections
import json
import pickle

import networkx as nx

import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from GraphData import GraphData


def main():
    data_path = "../../../GraphData/DS_all/"
    for db_name in ['NCI1', 'NCI109', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'DHFR', 'SYNTHETICnew']:
        # load the graph data
        graph_data = gdtgl.graph_data_to_graph_list(data_path, db_name, relabel_nodes=False)
        distances = []
        for graph in graph_data[0]:
            d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=2))
            # order the dictionary by the values
            d = collections.OrderedDict(sorted(d.items()))
            distances.append(d)
        # save list of dictionaries to a pickle file
        pickle.dump(distances, open(f"{db_name}_distances.pkl", 'wb'))
    for db_name in ['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']:
        # load the graph data
        graph_data = gdtgl.graph_data_to_graph_list(data_path, db_name, relabel_nodes=False)
        distances = []
        for graph in graph_data[0]:
            d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=2))
            # order the dictionary by the values
            d = collections.OrderedDict(sorted(d.items()))
            distances.append(d)
        # save list of dictionaries to a pickle file
        pickle.dump(distances, open(f"{db_name}_distances.pkl", 'wb'))


if __name__ == '__main__':
    main()