import collections
import json
import pickle

import networkx as nx

import ReadWriteGraphs.GraphDataToGraphList as gdtgl
from GraphData import GraphData

def save_distances(data_path="../../../GraphData/DS_all/", db_names=[], cutoff=6):
    max_distance = 0
    for db_name in db_names:
        # load the graph data
        graph_data = gdtgl.graph_data_to_graph_list(data_path, db_name, relabel_nodes=False)
        distances = []
        for graph in graph_data[0]:
            d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
            # order the dictionary by the values
            for _,value in d.items():
                for _, distance in value.items():
                    if distance > max_distance:
                        max_distance = distance
            distances.append(d)
        # save list of dictionaries to a pickle file
        pickle.dump(distances, open(f"{db_name}_distances.pkl", 'wb'))

def main():
    #save_distances(db_names=['NCI1', 'NCI109', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'DHFR', 'SYNTHETICnew'])
    #save_distances(db_names=[['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']], cutoff=2)
    save_distances(db_names=['MUTAG'], cutoff=None)



if __name__ == '__main__':
    main()