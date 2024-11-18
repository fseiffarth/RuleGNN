import torch

from src.utils.GraphData import NodeLabels


def load_labels(path='') -> NodeLabels:
    """
    Load the labels from a file.
    :param path: Path to the file
    :return: NodeLabels object
    """
    return NodeLabels(torch.load(path))