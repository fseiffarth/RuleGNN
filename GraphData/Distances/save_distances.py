import json
import pickle

from GraphData import GraphData


def main():
    data_path = "../../../GraphData/DS_all/"
    for db_name in ['NCI1', 'NCI109', 'Mutagenicity', 'DD', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'IMDB-MULTI',
                    'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'DHFR', 'SYNTHETICnew', 'COLLAB']:
        # load the graph data
        graph_data = GraphData.GraphData()
        graph_data.init_from_graph_db(data_path, db_name, with_distances=True, with_cycles=False,
                                      relabel_nodes=True, use_features=False, use_attributes=False)

        # save the distances to a file
        distances = graph_data.distance_list
        # save list of dictionaries to a pickle file
        pickle.dump(distances, open(f"{db_name}_distances.pkl", 'wb'))


if __name__ == '__main__':
    main()