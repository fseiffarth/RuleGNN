from src.utils.GraphData import NodeLabels


def load_labels(path='') -> NodeLabels:
    node_labels = NodeLabels()
    """
    Load the labels from a file.

    :param file: str
    :return: list of lists
    """

    with open(path, 'r') as f:
        labels = f.read().splitlines()
        labels = [list(map(int, l.split())) for l in labels]
    node_labels.node_labels = labels
    node_labels.db_unique_node_labels = {}
    node_labels.unique_node_labels = []
    # set db_unique_node_labels
    for g_labels in labels:
        node_labels.unique_node_labels.append({})
        for l in g_labels:
            if l not in node_labels.db_unique_node_labels:
                node_labels.db_unique_node_labels[l] = 1
            else:
                node_labels.db_unique_node_labels[l] += 1
            if l not in node_labels.unique_node_labels[-1]:
                node_labels.unique_node_labels[-1][l] = 1
            else:
                node_labels.unique_node_labels[-1][l] += 1
    node_labels.num_unique_node_labels = len(node_labels.db_unique_node_labels)

    return node_labels