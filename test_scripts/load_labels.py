from pathlib import Path

import torch


def load_label_files(path: Path):
    """
    Load the label files from the path and return a dictionary with the labels
    :param path: path to the label files
    :return: dictionary with the labels
    """
    labels = torch.load(path)
    return labels


def main():
    base_path = Path('/home/florian/Documents/Code/GNNs/RuleGNN/ReproduceExtended/Data/Labels/ZINC/')
    path_20 = base_path.joinpath('ZINC_labels_simple_cycles_20_primary.pt')
    path_10 = base_path.joinpath('ZINC_labels_simple_cycles_10_primary.pt')
    path_50 = base_path.joinpath('ZINC_labels_simple_cycles_50_primary.pt')
    labels_10 = load_label_files(path_10)
    labels_20 = load_label_files(path_20)
    labels_50 = load_label_files(path_50)
    print(labels)


if __name__ == '__main__':
    main()