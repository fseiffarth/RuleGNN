# Create your own graph db in th correct format
from typing import List

import networkx as nx


def save_graph_db(path: str, graph_db_name: str, graphs: List[nx.Graph], labels: List[int]):
    with open(f'{path}/{graph_db_name.txt}', "w") as f:
        # write number of graphs in the first line
        f.write(str(len(graphs)) + "\n")
        # write graphs labels separated by a space
        f.write(" ".join(map(str, labels)) + "\n")
        # iterate over graphs and use nx export to write them to the file with data in the correct format
        for graph in graphs:
            nx.write_adjlist(graph, f)
            # add a line break to separate the graphs
            f.write("\n")

# Load your own graph db in the correct format
def load_graph_db(path: str, graph_db_name: str) -> List[nx.Graph]:
    with open(f'{path}/{graph_db_name.txt}', "r") as f:
        # read number of graphs
        num_graphs = int(f.readline())
        # read graph labels
        labels = list(map(int, f.readline().split()))
        graphs = []
        # iterate over the number of graphs and read them using nx read_adjlist
        for _ in range(num_graphs):
            graph = nx.read_adjlist(f)
            graphs.append(graph)
    return (graphs, labels, [], graph_db_name)