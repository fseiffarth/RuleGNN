import sys

import networkx as nx
import numpy as np

from src.utils.GraphData import ShareGNNDataset


class EditPath():
    def __init__(self, db_name=None, start_id=None, end_id=None, start_graph:nx.Graph=None, end_graph:nx.Graph=None, edit_path=None):
        if edit_path is not None:
            # node operations
            self.distance = edit_path[2]
            node_operations_all = [(a, b) for (a, b) in edit_path[0] if a != b]
            self.node_operations = dict()
            self.node_operations['remove'] = [a for (a, b) in node_operations_all if b is None]
            self.node_operations['add'] = [b for (a, b) in node_operations_all if a is None]
            if nx.get_node_attributes(start_graph, 'primary_label') and nx.get_node_attributes(end_graph, 'primary_label'):
                self.node_operations['relabel'] = [(a, b) for (a, b) in node_operations_all if a is not None and b is not None and start_graph.nodes[a]['primary_label'] != end_graph.nodes[b]['primary_label']]
            self.node_map = dict()
            for (a, b) in node_operations_all:
                if a is not None and b is not None:
                    self.node_map[a] = b
            # edge operations
            edge_operations_all = [(a, b) for (a, b) in edit_path[1] if a != b]
            self.edge_operations = dict()
            self.edge_operations['remove'] = [a for (a, b) in edge_operations_all if b is None]
            self.edge_operations['add'] = [b for (a, b) in edge_operations_all if a is None]
            # check wheter 'label' is in the edge attributes
            if nx.get_edge_attributes(start_graph, 'label') and nx.get_edge_attributes(end_graph, 'label'):
                self.edge_operations['relabel'] = [(a, b) for (a, b) in edge_operations_all if a is not None and b is not None and end_graph.edges[a]['label'] != end_graph.edges[b]['label']]
            self.all_operations = list()
            for key, value in self.node_operations.items():
                self.all_operations.extend([(f'{key}_node', op) for op in value])
            for key, value in self.edge_operations.items():
                self.all_operations.extend([(f'{key}_edge', op) for op in value])


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
            'edge_operations': self.edge_operations,
            'all_operations': self.all_operations
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
        if 'all_operations' in json_obj:
            self.all_operations = json_obj['all_operations']
        else:
            self.all_operations = list()
            for key, value in self.node_operations.items():
                self.all_operations.extend([(f'{key}_node', op) for op in value])
            for key, value in self.edge_operations.items():
                self.all_operations.extend([(f'{key}_edge', op) for op in value])
        return self


    def create_edit_path_graphs(self, nx_graph1, nx_graph2, seed=42, plotting=True):
        """
        Create a sequence of networkx graphs representing the edit path. Starting from nx_graph1, applying the node and edge operations
        and ending with nx_graph2.
        """
        graph_sequence = [nx_graph1]
        if plotting:
            plot_graph(graph_sequence[-1])
        # create a shuffled list of operations to apply
        shuffled_operations = self.all_operations.copy()
        np.random.seed(seed)
        np.random.shuffle(shuffled_operations)
        unsuccessful_operations = list()
        for op_type, op_value in shuffled_operations:
            # create a copy of the last graph in the sequence
            last_graph = graph_sequence[-1].copy()
            # differentiate between operation types
            if op_type == 'add_node':
                # add a node with the given value
                last_graph.add_node(op_value, primary_label=op_value)
                graph_sequence.append(last_graph)
                if plotting:
                    plot_graph_changes(graph_sequence[-1], node=op_value, type='add')
            elif op_type == 'remove_node':
                # remove the node with the given value
                if last_graph.has_node(op_value):
                    last_graph.remove_node(op_value)
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], node=op_value, type='remove')
            elif op_type == 'change_node':
                pass
            elif op_type == 'add_edge':
                # add an edge between the two nodes with the given values
                if last_graph.has_node(op_value[0]) and last_graph.has_node(op_value[1]):
                    last_graph.add_edge(op_value[0], op_value[1])
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], edge=op_value, type='add')
                    pass
            elif op_type == 'remove_edge':
                # remove the edge between the two nodes with the given values
                if last_graph.has_edge(op_value[0], op_value[1]):
                    last_graph.remove_edge(op_value[0], op_value[1])
                    graph_sequence.append(last_graph)
                    if plotting:
                        plot_graph_changes(graph_sequence[-1], edge=op_value, type='remove')
                    pass
            elif op_type == 'change_edge':
                pass




            pass
        # TODO: add the node operations to the graph1 and graph2
        return graph_sequence

def plot_graph(nx_graph: nx.Graph, with_node_ids: bool = False):
    """
    Plot the given networkx graph.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(nx_graph)
    nx.draw_networkx_nodes(nx_graph, pos, node_size=700, node_color='lightblue')
    node_labels = nx.get_node_attributes(nx_graph, 'primary_label')
    node_ids = [node_id for node_id in nx_graph.nodes()] if with_node_ids else None
    if node_ids is not None:
        node_labels = {node_id: f"{node_labels[node_id]} ({node_id})" for node_id in node_ids if node_id in node_labels}
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)
    nx.draw_networkx_edges(nx_graph, pos)
    plt.show()

def plot_graph_changes(nx_graph: nx.Graph, edge=None, node=None, type=None):
    """
    Plot the given networkx graph.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    pos = nx.kamada_kawai_layout(nx_graph)
    node_colors = ['lightblue'] * nx_graph.number_of_nodes()
    # if and edge is removed, color the endpoint nodes red
    if edge is not None and (type == 'remove' or type == 'add'):
            node_colors[edge[0]] = 'red'
            node_colors[edge[1]] = 'red'
    if node is not None and (type == 'remove' or type == 'add'):
        node_colors[node] = 'red'
    nx.draw_networkx_nodes(nx_graph, pos, node_size=700, node_color=node_colors)
    node_labels = nx.get_node_attributes(nx_graph, 'primary_label')
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=12)
    nx.draw_networkx_edges(nx_graph, pos)
    plt.show()

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


def generate_pairwise_optimal_paths(share_dataset:ShareGNNDataset, output_dir:str = 'data/'):
    """
    Generates pairwise optimal paths for the given database.

    Args:
        share_dataset: ShareGNNDataset object containing the dataset.
        output_dir: Directory where the edit paths will be saved.
    """
    # Load the Mutag dataset with the ShareGNNDataset class
    share_dataset.create_nx_graphs()
    print(f"Loaded MUTAG dataset with {len(share_dataset)} graphs")

    def node_match_primary(n1, n2):
        return n1['primary_label'] == n2['primary_label']

    num_max_edit_paths_per_pair = 1
    nx_graphs = share_dataset.nx_graphs
    # plot the first graph
    plot_graph(nx_graphs[0], with_node_ids=True)
    # plot the second graph
    plot_graph(nx_graphs[1], with_node_ids=True)
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
                optimal_edit_paths.append(EditPath(db_name, i, j, start_graph=nx_graphs[i], end_graph=nx_graphs[j], edit_path=x))
                print(f"Calculated optimal edit path {p+1} / {num_max_edit_paths_per_pair} for graphs {i} and {j}")
                p += 1
            global_edit_paths[(i, j)] = optimal_edit_paths
    save_edit_path_to_file(db_name, global_edit_paths, file_path=output_dir)
    return


if __name__ == '__main__':
    db_name = 'MUTAG'
    # Load the Mutag dataset with the ShareGNNDataset class
    share_dataset = ShareGNNDataset(
        root='data',
        name=db_name,
        from_existing_data='TUDataset',
        task='graph_classification'
    )
    generate_pairwise_optimal_paths(share_dataset, output_dir='data/')

    share_dataset.create_nx_graphs()
    nx_graphs = share_dataset.nx_graphs
    edit_paths = load_edit_paths_from_file(db_name=db_name, file_path='data/')
    edit_paths[(0, 1)][0].create_edit_path_graphs(nx_graphs[0], nx_graphs[1])
    pass
