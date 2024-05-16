import os

import networkx as nx

from GraphData import GraphData
from GraphData.Distances.load_distances import load_distances

from GraphBenchmarks.csl import CSL
from GraphData.GraphData import get_graph_data


def main():
    # load the graph data
    graph_db = 'REDDIT-BINARY'
    #graph_db = 'Snowflakes'
    data_path = '../GraphData/DS_all/'
    #data_path = 'GraphBenchmarks/Data/'
    distance_path = "GraphData/Distances/"
    """
    Create Input data, information and labels from the graphs for training and testing
    """
    graph_data = get_graph_data(graph_db, data_path, distance_path, use_features=True, use_attributes=False)
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

    # get largest graph
    print(f'Max nodes: {max(node_numbers)}')
    # get max diameter
    print(f'Max diameter: {max(diameters)}')
    node_numbers.sort()
    # ge the 0.6 and 0.9 median
    print(f'Median 0.6: {node_numbers[int(0.6 * len(node_numbers))]}')
    print(f'Median 0.9: {node_numbers[int(0.9 * len(node_numbers))]}')



if __name__ == "__main__":
    main()