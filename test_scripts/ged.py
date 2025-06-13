import networkx as nx
import matplotlib.pyplot as plt
from src.utils.GraphData import ShareGNNDataset

class EditPath():
    def __init__(self, db_name=None, start_id=None, end_id=None, edit_path=None):
        if edit_path is not None:
            # node operations
            self.distance = edit_path[2]
            node_operations_all = [(a, b) for (a, b) in edit_path[0] if a != b]
            self.node_operations = dict()
            self.node_operations['remove'] = [a for (a, b) in node_operations_all if b is None]
            self.node_operations['add'] = [b for (a, b) in node_operations_all if a is None]
            self.node_operations['change'] = [(a, b) for (a, b) in node_operations_all if a is not None and b is not None]
            # edge operations
            edge_operations_all = [(a, b) for (a, b) in edit_path[1] if a != b]
            self.edge_operations = dict()
            self.edge_operations['remove'] = [a for (a, b) in edge_operations_all if b is None]
            self.edge_operations['add'] = [b for (a, b) in edge_operations_all if a is None]
            self.edge_operations['change'] = [(a, b) for (a, b) in edge_operations_all if a is not None and b is not None]
            self.db_name = db_name
            self.start_id = start_id
            self.end_id = end_id

    # serialize the class to a json object
    def toJSON(self):
        return {
            'db_name': self.db_name,
            'start_id': self.start_id,
            'end_id': self.end_id,
            'distance': self.distance,
            'node_operations': self.node_operations,
            'edge_operations': self.edge_operations
        }
    def loadJSON(self, json_obj):
        """
        Load the edit path from a JSON object.
        """
        self.db_name = json_obj['db_name']
        self.start_id = json_obj['start_id']
        self.end_id = json_obj['end_id']
        self.distance = json_obj['distance']
        self.node_operations = json_obj['node_operations']
        self.edge_operations = json_obj['edge_operations']
        return self

    def create_edit_path_graphs(self, nx_graph1, nx_graph2):
        """
        Create two NetworkX graphs representing the edit path.
        """
        # Create a copy of the original graphs
        graph1 = nx_graph1.copy()
        graph2 = nx_graph2.copy()
        # TODO: add the node operations to the graph1 and graph2

def save_edit_path_to_file(db_name, edit_paths, file_path):
    # save the global edit paths to a file
    with open(f'{file_path}/{db_name}_ged_paths.paths', 'w') as f:
        for i in range(len(edit_paths)):
            for j in range(i + 1, len(edit_paths)):
                if (i, j) not in edit_paths:
                    continue
                else:
                    for edit_path in edit_paths[(i, j)]:
                        f.write(f"{i} {j} {edit_path.toJSON()}\n")

def load_edit_paths_from_file(db_name, file_path):
    # load the global edit paths from a file
    edit_paths = dict()
    with open(f'{file_path}/{db_name}_ged_paths.paths', 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            if len(parts) < 3:
                continue
            i, j, json_str = int(parts[0]), int(parts[1]), parts[2]
            edit_path = EditPath().loadJSON(eval(json_str))
            if (i, j) not in edit_paths:
                edit_paths[(i, j)] = []
            edit_paths[(i, j)].append(edit_path)
    return edit_paths

def optimal_edit_path():
    db_name = 'MUTAG'
    # Load the Mutag dataset with the ShareGNNDataset class
    share_dataset = ShareGNNDataset(
        root='data',
        name=db_name,
        from_existing_data='TUDataset',
        task='graph_classification'
    )
    share_dataset.create_nx_graphs()
    print(f"Loaded MUTAG dataset with {len(share_dataset)} graphs")

    def node_match_primary(n1, n2):
        return n1['primary_label'] == n2['primary_label']

    num_max_edit_paths_per_pair = 1
    nx_graphs = share_dataset.nx_graphs[:3]
    # iterate over all the graph pairs
    global_edit_paths = dict()
    for i in range(len(nx_graphs)):
        for j in range(i + 1, len(nx_graphs)):
            print(f"Comparing graph {i} with graph {j}")
            result_nx = nx.optimize_edit_paths(nx_graphs[i], nx_graphs[j], node_match=node_match_primary)
            optimal_edit_paths = []
            p = 0
            while p < num_max_edit_paths_per_pair:
                x = next(result_nx)
                optimal_edit_paths.append(EditPath(db_name, i, j, x))
                print(f"Calculated optimal edit path {p+1} / {num_max_edit_paths_per_pair} for graphs {i} and {j}")
                p += 1
            global_edit_paths[(i, j)] = optimal_edit_paths
    save_edit_path_to_file(db_name, global_edit_paths, 'data')
    # load the edit paths from the file
    load_edit_paths_from_file(db_name, 'data')




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
    # create graphs on the optimal edit path steps
    node_changes = optimal_edit_path[0]
    edge_changes = optimal_edit_path[1]

    pass



if __name__ == "__main__":
    optimal_edit_path()
