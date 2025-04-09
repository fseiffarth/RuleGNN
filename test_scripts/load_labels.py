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
    test_0 = load_label_files('Labels/MUTAG_labels_wl_labeled_simple_cycles_6_6_base_labels_1.pt')
    test_1 = load_label_files('Labels/MUTAG_labels_wl_labeled_simple_cycles_6_6_primary_base_labels_1.pt')
    base_path = Path('/home/florian/Documents/Code/GNNs/RuleGNN/ReproduceExtended/Data/Labels/ZINC/')
    base_path_ring = Path('/home/florian/Documents/Code/GNNs/RuleGNN/ReproduceExtended/Data/Labels/RingCounting3/')
    labels = load_label_files(base_path_ring.joinpath('RingCounting3_labels_subgraph_0.pt'))
    labels2 = load_label_files(base_path_ring.joinpath('RingCounting3_labels_induced_cycles_3.pt'))

    path_20 = base_path.joinpath('ZINC_labels_simple_cycles_20_primary.pt')
    path_10 = base_path.joinpath('ZINC_labels_simple_cycles_10_primary.pt')
    path_50 = base_path.joinpath('ZINC_labels_simple_cycles_50_primary.pt')
    labels_10 = load_label_files(path_10)
    labels_20 = load_label_files(path_20)
    labels_50 = load_label_files(path_50)
    print(labels)


if __name__ == '__main__':
    main()