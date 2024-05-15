import os

import networkx as nx

from GraphData import GraphData
from GraphData.Distances.load_distances import load_distances

from GraphBenchmarks.csl import CSL


def main():
    # load the graph data
    graph_db = 'REDDIT-BINARY'
    data_path = '../GraphData/DS_all/'
    distance_path = "GraphData/Distances/"
    """
    Create Input data, information and labels from the graphs for training and testing
    """
    if graph_db == "CSL":
        csl = CSL()
        graph_data = csl.get_graphs(with_distances=False)
        if os.path.isfile(f'{distance_path}{graph_db}_distances.pkl'):
            distance_list = load_distances(db_name=graph_db,
                                           path=f'{distance_path}{graph_db}_distances.pkl')
            graph_data.distance_list = distance_list
        node_numbers = []
        for graph in graph_data.graphs:
            node_numbers.append(graph.number_of_nodes())
        node_numbers.sort()
        # ge the 0.6 and 0.9 median
        print(node_numbers[int(0.6 * len(node_numbers))])
        print(node_numbers[int(0.9 * len(node_numbers))])
    else:
        graph_data = GraphData.GraphData()
        graph_data.init_from_graph_db(data_path, graph_db, with_distances=False, with_cycles=False,
                                      relabel_nodes=True, use_features=False,
                                      use_attributes=False,
                                      distances_path=distance_path)
        # get a list of node numbers sorted by the number of nodes in each graph
        node_numbers = []
        diameters = []
        for graph in graph_data.graphs:
            node_numbers.append(graph.number_of_nodes())
            # get all connected components
            connected_components = nx.connected_components(graph)
            for connected_component in connected_components:
                # get the diameter of the connected component
                subgraph = graph.subgraph(connected_component)
                diameter = nx.diameter(subgraph)
                diameters.append(diameter)

        # get max diameter
        print(max(diameters))
        node_numbers.sort()
        # ge the 0.6 and 0.9 median
        print(node_numbers[int(0.6 * len(node_numbers))])
        print(node_numbers[int(0.9 * len(node_numbers))])



if __name__ == "__main__":
    main()