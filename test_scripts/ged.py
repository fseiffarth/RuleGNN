import networkx as nx
import matplotlib.pyplot as plt
from src.utils.GraphData import ShareGNNDataset
def optimal_edit_path():
    # Load the Mutag dataset with the ShareGNNDataset class
    share_dataset = ShareGNNDataset(
        root='data',
        name='MUTAG',
        from_existing_data='TUDataset',
        task='graph_classification'
    )
    print(f"Loaded MUTAG dataset with {len(share_dataset)} graphs")

    # Convert ShareGNNDataset to gklearn.utils.Dataset
    share_dataset.create_nx_graphs(directed=False)
    G2 = nx.path_graph(10)
    G1 = nx.path_graph(3)
    # add node labels 0,1,0 to G1
    G1.nodes[0]['label'] = 0
    G1.nodes[1]['label'] = 0
    G1.nodes[2]['label'] = 1

    # add node labels 0,1,0,1,0,1,0,1,0,1 to G2
    node_labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    for i, node in enumerate(G2.nodes):
        G2.nodes[node]['label'] = node_labels[i]

    def node_match(n1, n2):
        return n1['label'] == n2['label']

    def node_match_primary(n1, n2):
        return n1['primary_label'] == n2['primary_label']

    result = nx.optimize_edit_paths(G1, G2, node_match=node_match)
    for x in result:
        print(x)
        break

    nx_graphs = share_dataset.nx_graphs
    NX1 = nx_graphs[0]
    NX2 = nx_graphs[1]
    # plot the graphs using kawai layout
    pos_n1 = nx.layout.kamada_kawai_layout(NX1)
    pos_n2 = nx.layout.kamada_kawai_layout(NX2)

    nx.draw_networkx_nodes(NX1, pos_n1, node_color='blue', label='NX1')
    nx.draw_networkx_edges(NX1, pos_n1, edge_color='blue')
    nx.draw_networkx_labels(NX1, pos_n1, labels=nx.get_node_attributes(NX1, 'primary_label'), font_color='white')
    plt.show()
    plt.clf()
    nx.draw_networkx_nodes(NX2, pos_n2, node_color='red', label='NX2')
    nx.draw_networkx_edges(NX2, pos_n2, edge_color='red')
    nx.draw_networkx_labels(NX2, pos_n2, labels=nx.get_node_attributes(NX2, 'primary_label'), font_color='white')
    plt.show()

    nodes_nx1 = NX1.nodes(data=True)
    print(f"Nodes in NX1: {nodes_nx1}")
    if NX1.number_of_nodes() < NX2.number_of_nodes():
        print("NX1 has fewer nodes than NX2")
        result_nx = nx.optimize_edit_paths(NX1, NX2, node_match=node_match_primary)
    else:
        print("NX2 has fewer or equal nodes than NX1")
        result_nx = nx.optimize_edit_paths(NX2, NX1, node_match=node_match_primary)
    optimal_edit_path = None
    for x in result_nx:
        optimal_edit_path = x
        print(x)
        break


    pass



if __name__ == "__main__":
    optimal_edit_path()
