import pickle

import networkx as nx

from src.utils.GraphData import get_graph_data


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


def save_distances(data_path="../../../BenchmarkGraphs/DS_all/", db_names=[], cutoff=None, distance_path=""):
    for db_name in db_names:
        graph_data = get_graph_data(db_name=db_name, data_path=data_path, with_distances=False)
        write_distances(graph_data=graph_data, db_name=db_name, cutoff=cutoff, distance_path=distance_path)
        # load the graph data
        # if db_name == 'CSL_original':
        #     csl = CSL_original()
        #     graph_data = csl.get_graphs(with_distances=False)
        #     max_distance = write_distances(graph_data, db_name, cutoff)
        # elif db_name == 'ZINC_original':
        #     zinc_train = ZINC_original(root="../ZINC_original/", subset=True, split='train')
        #     zinc_val = ZINC_original(root="../ZINC_original/", subset=True, split='val')
        #     zinc_test = ZINC_original(root="../ZINC_original/", subset=True, split='test')
        #     graph_data = zinc_to_graph_data(zinc_train, zinc_val, zinc_test, "ZINC_original")
        #     max_distance = write_distances(graph_data, db_name, cutoff)
        # elif db_name == 'LongRings':
        #     graph_data = BenchmarkGraphs.BenchmarkGraphs()
        #     graph_data.load_from_benchmark("LongRings", "BenchmarkGraphs/")
        #     max_distance = write_distances(graph_data, db_name, cutoff, distance_path)
        # else:
        #     graph_data = gdtgl.graph_data_to_graph_list(data_path, db_name, relabel_nodes=False)
        #     graph_data = graph_data[0]
        #     max_distance = write_distances(graph_data, db_name, cutoff, distance_path)


def main():
    #save_distances(db_names=['NCI1', 'NCI109', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'DHFR', 'SYNTHETICnew'])
    #save_distances(db_names=[['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']], cutoff=2)
    #save_distances(db_names=['LongRings'], cutoff=None)
    save_distances(db_names=['DHFR'], cutoff=None)

if __name__ == '__main__':
    main()
