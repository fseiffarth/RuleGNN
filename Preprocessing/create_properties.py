import gzip
import os
import pickle

import networkx as nx
import yaml

from utils.GraphData import get_graph_data
from utils.load_labels import load_labels
import copy

from utils.utils import convert_to_list


def write_distance_properties(data_path, db_name, cutoff, out_path="") -> None:
    out = f"{out_path}{db_name}_distances.prop"
    # check if the file already exists and if not create it
    if not os.path.exists(out):
        graph_data = get_graph_data(db_name=db_name, data_path=data_path)
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
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": list(valid_properties), 'description': 'Distance',
                                 'list_of_values': f'{list(valid_properties)}'}
        with open(f"{out_path}{db_name}_distances.yml", 'w') as f:
            yaml.dump(valid_properties_dict, f)


def write_distance_circle_properties(data_path, label_path, db_name, cutoff, out_path="") -> None:
    out = f"{out_path}{db_name}_circle_distances.prop"
    # check if the file already exists and if not create it
    if not os.path.exists(out):
        graph_data = get_graph_data(db_name=db_name, data_path=data_path)
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
                for (i, j) in value:
                    label_i = circle_labels.node_labels[graph_id][i]
                    label_j = circle_labels.node_labels[graph_id][j]
                    # determine the final label
                    final_label = (key, label_i, label_j)
                    if final_label in final_dict:
                        final_dict[final_label].append((i, j))
                    else:
                        final_dict[final_label] = [(i, j)]
                    valid_properties.add(final_label)
            final_properties.append(final_dict)

        # sort valid properties by tuple 1,2,3 entries
        valid_properties = sorted(valid_properties, key=lambda x: (x[0], x[1], x[2]))
        # save list of dictionaries to a pickle file
        pickle_data = pickle.dumps(final_properties)

        # compress with gzip
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))
        v_properties = [f'{convert_to_list(x)}' for x in valid_properties]
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": list(v_properties), 'description': 'Distance, In cycle -> In cycle',
                                 'list_of_values': f'{valid_properties}'}
        with open(f"{out_path}{db_name}_circle_distances.yml", 'w') as f:
            yaml.dump(valid_properties_dict, f)


def write_distance_edge_properties(data_path, db_name, out_path="") -> None:
    out = f"{out_path}{db_name}_edge_label_distances.prop"
    # check if the file already exists and if not create it
    if not os.path.exists(out):
        graph_data = get_graph_data(db_name=db_name, data_path=data_path, use_features=True)
        distances = []
        valid_properties = set()
        final_properties = []
        for graph in graph_data.graphs:
            graph_map = {}
            d = dict(nx.all_pairs_all_shortest_paths(graph))
            # replace the end nodes with the label of the edge between them
            # copy d
            d_edges = copy.deepcopy(d)
            for key, value in d.items():
                for key2, value2 in value.items():
                    for path_id, shortest_path in enumerate(value2):
                        edge_label_sequence = []
                        if len(shortest_path) == 1:
                            d_edges[key][key2][path_id] = []
                        else:
                            for i in range(0, len(shortest_path) - 1):
                                edge_start = shortest_path[i]
                                edge_end = shortest_path[i + 1]
                                # get the label of the edge
                                edge_label = graph[edge_start][edge_end]['label']
                                if len(edge_label) == 1:
                                    edge_label_sequence.append(int(edge_label[0]))
                                else:
                                    raise ValueError("Edge label is not 1-dimensional.")
                            d_edges[key][key2][path_id] = edge_label_sequence
            for start_node in graph.nodes:
                for end_node in graph.nodes:
                    if start_node in d_edges and end_node in d_edges[start_node]:
                        paths = d[start_node][end_node]
                        paths_labels = d_edges[start_node][end_node]
                        distance = len(paths[0]) - 1
                        number_of_paths = len(paths)
                        label_occurrences = []
                        for path in paths_labels:
                            for label in path:
                                label = int(label)
                                while len(label_occurrences) <= label:
                                    label_occurrences.append(0)
                                label_occurrences[label] += 1
                        label_tuple = (distance, number_of_paths, tuple(label_occurrences))
                        if label_tuple in graph_map:
                            graph_map[label_tuple].append((start_node, end_node))
                        else:
                            graph_map[label_tuple] = [(start_node, end_node)]
                        valid_properties.add(label_tuple)

            final_properties.append(graph_map)

        # sort valid properties by tuple 1,2,3 entries
        valid_properties = sorted(valid_properties, key=lambda x: (x[0], x[1], x[2]))
        # save list of dictionaries to a pickle file
        pickle_data = pickle.dumps(final_properties)

        # compress with gzip
        with open(out, 'wb') as f:
            f.write(gzip.compress(pickle_data))

        # create a dictionary of valid properties
        v_values = []
        list_of_values = []
        list_of_values_str = ''
        for value in valid_properties:
            value = convert_to_list(value)
            v_values.append(f"{value}")
            list_of_values.append(value)
        list_of_values = f'{list_of_values}'
        # save an additional .info file that stores the set of valid_properties as a yml file
        valid_properties_dict = {"valid_values": v_values,
                                 "description": "Distance, Path number, Edge label occurrences",
                                 "list_of_values": list_of_values}
        # save an additional .info file that stores the set of valid_properties as a yml file
        with open(f"{out_path}{db_name}_edge_label_distances.yml", 'w') as f:
            yaml.dump(valid_properties_dict, f)


def main():
    data_path = "../GraphData/DS_all/"
    label_path = "Data/Labels/"

    ### Test Dataset Properties

    # ZINC
    write_distance_edge_properties(data_path="Data/BenchmarkGraphs/", db_name='ZINC', out_path="Data/Properties/")
    write_distance_circle_properties(data_path="Data/BenchmarkGraphs/", label_path=label_path, db_name='ZINC', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path="Data/BenchmarkGraphs/", db_name='ZINC', cutoff=None, out_path="Data/Properties/")

    # MUTAG
    write_distance_edge_properties(data_path=data_path, db_name='MUTAG', out_path="Data/Properties/")
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='MUTAG', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='MUTAG', cutoff=None, out_path="Data/Properties/")



    # PTC (FM, FR, MM, MR)
    write_distance_edge_properties(data_path=data_path, db_name='PTC_FM', out_path="Data/Properties/")
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='PTC_FM', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='PTC_FM', cutoff=None, out_path="Data/Properties/")
    write_distance_edge_properties(data_path=data_path, db_name='PTC_FR', out_path="Data/Properties/")
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='PTC_FR', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='PTC_FR', cutoff=None, out_path="Data/Properties/")
    write_distance_edge_properties(data_path=data_path, db_name='PTC_MM', out_path="Data/Properties/")
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='PTC_MM', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='PTC_MM', cutoff=None, out_path="Data/Properties/")
    write_distance_edge_properties(data_path=data_path, db_name='PTC_MR', out_path="Data/Properties/")
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='PTC_MR', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='PTC_MR', cutoff=None, out_path="Data/Properties/")

    # Paper Dataset Properties
    # DHFR
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='DHFR', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='DHFR', cutoff=None, out_path="Data/Properties/")

    # NCI1
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='NCI1', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='NCI1', cutoff=None, out_path="Data/Properties/")

    # NCI109
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='NCI109', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='NCI109', cutoff=None, out_path="Data/Properties/")

    # Mutagenicity
    write_distance_circle_properties(data_path=data_path, label_path=label_path, db_name='Mutagenicity', cutoff=None,
                                     out_path="Data/Properties/")
    write_distance_properties(data_path=data_path, db_name='Mutagenicity', cutoff=None, out_path="Data/Properties/")

    # IMDB-BINARY
    write_distance_properties(data_path=data_path, db_name='IMDB-BINARY', cutoff=None, out_path="Data/Properties/")

    # IMDB-MULTI
    write_distance_properties(data_path=data_path, db_name='IMDB-MULTI', cutoff=None, out_path="Data/Properties/")


if __name__ == '__main__':
    main()
