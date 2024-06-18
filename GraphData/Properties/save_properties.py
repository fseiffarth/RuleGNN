import gzip
import pickle

import networkx as nx
import yaml

from GraphData.GraphData import get_graph_data


def write_distance_properties(graph_data, db_name, cutoff, path="") -> None:
    distances = []
    valid_properties = set()
    for graph in graph_data.graphs:
        d = dict(nx.all_pairs_shortest_path_length(graph, cutoff=cutoff))
        # use d to make a dictionary of pairs for each distance
        new_d = {}
        for key, value in d.items():
            for key2, value2 in value.items():
                if value2 in new_d:
                    new_d[value2].append((key, key2))
                else:
                    new_d[value2] = [(key, key2)]
            pass
        distances.append(new_d)
        for key in new_d.keys():
            valid_properties.add(key)
    # save list of dictionaries to a pickle file
    pickle_data = pickle.dumps(distances)
    # compress with gzip
    with open(f"{path}{db_name}_distances.prop", 'wb') as f:
        f.write(gzip.compress(pickle_data))
    # save an additional .info file that stores the set of valid_properties as a yml file
    valid_properties_dict = {"valid_values": list(valid_properties)}
    with open(f"{path}{db_name}_distances.yml", 'w') as f:
        yaml.dump(valid_properties_dict, f)


def save_properties(data_path="../../../GraphData/DS_all/", db_names=[], cutoff=None, path="Data/"):
    for db_name in db_names:
        graph_data = get_graph_data(db_name=db_name, data_path=data_path, with_distances=False)
        write_distance_properties(graph_data=graph_data, db_name=db_name, cutoff=cutoff, path=path)


def main():
    #save_distances(db_names=['NCI1', 'NCI109', 'Mutagenicity', 'IMDB-BINARY', 'IMDB-MULTI', 'PROTEINS', 'ENZYMES', 'DHFR', 'SYNTHETICnew'])
    #save_distances(db_names=[['DD', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'COLLAB']], cutoff=2)
    #save_distances(db_names=['LongRings'], cutoff=None)
    save_properties(db_names=['DHFR'], cutoff=None)

if __name__ == '__main__':
    main()
