import gzip
import pickle

import networkx as nx
import yaml

from GraphData.GraphData import get_graph_data
from GraphData.Labels.generator.load_labels import load_labels


def write_distance_properties(data_path, db_name, cutoff, out_path="") -> None:
    graph_data = get_graph_data(db_name=db_name, data_path=data_path, with_distances=False)
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
    with open(f"{out_path}{db_name}_distances.prop", 'wb') as f:
        f.write(gzip.compress(pickle_data))
    # save an additional .info file that stores the set of valid_properties as a yml file
    valid_properties_dict = {"valid_values": list(valid_properties)}
    with open(f"{out_path}{db_name}_distances.yml", 'w') as f:
        yaml.dump(valid_properties_dict, f)


def write_distance_circle_properties(data_path, label_path, db_name, cutoff, out_path="") -> None:
    graph_data = get_graph_data(db_name=db_name, data_path=data_path, with_distances=False)
    distances = []
    circle_labels = load_labels(f"{label_path}{db_name}_cycles_20_labels.txt")
    label_combinations = circle_labels.num_unique_node_labels ** 2
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

    final_properties = []
    valid_properties.clear()

    for graph_id, graph in enumerate(graph_data.graphs):
        final_dict = {}
        for key, value in distances[graph_id].items():
            for (i,j) in value:
                label_i = circle_labels.node_labels[graph_id][i]
                label_j = circle_labels.node_labels[graph_id][j]
                # determine the final label
                pos = 0
                if label_i == 0 and label_j == 0:
                    pos = 0
                elif label_i == 0 and label_j == 1:
                    pos = 1
                elif label_i == 1 and label_j == 0:
                    pos = 2
                else:
                    pos = 3
                final_label = key * label_combinations + pos
                if final_label in final_dict:
                    final_dict[final_label].append((i,j))
                else:
                    final_dict[final_label] = [(i,j)]
        final_properties.append(final_dict)
        for key in final_dict.keys():
            valid_properties.add(key)



    # save list of dictionaries to a pickle file
    pickle_data = pickle.dumps(final_properties)
    # compress with gzip
    with open(f"{out_path}{db_name}_circle_distances.prop", 'wb') as f:
        f.write(gzip.compress(pickle_data))
    # save an additional .info file that stores the set of valid_properties as a yml file
    valid_properties_dict = {"valid_values": list(valid_properties)}
    with open(f"{out_path}{db_name}_circle_distances.yml", 'w') as f:
        yaml.dump(valid_properties_dict, f)



def main():
    data_path = "../../../GraphData/DS_all/"
    label_path = "../../GraphData/Labels/"
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='DHFR', cutoff=None, out_path="Data/")
    write_distance_properties(data_path=data_path, db_name='DHFR', cutoff=None, out_path="Data/")



if __name__ == '__main__':
    main()
