import sys
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
    db = 'ZINC'
    # get work directory
    work_dir = Path(sys.argv[0]).resolve().parent.parent
    label_root_path = Path(f'{work_dir}/paper_experiments/Data/Labels/{db}/')
    # get all files in the label root path
    label_files = list(label_root_path.glob(f'{db}_labels_*.pt'))
    label_data = {}
    for file in label_files:
        # get name of the file without the extension
        file_name = file.stem
        label_data[file_name] = load_label_files(file)

    # create statistics over the label data as table name, number of unique labels
    label_statistics = []
    for name, labels in label_data.items():
        num_unique_labels = len(torch.unique(labels[2][:,1]))
        label_statistics.append({
            'name': name,
            'num_unique_labels': num_unique_labels
        })
    print("Label Statistics:")
    for stat in label_statistics:
        print(f"{stat['name']}: {stat['num_unique_labels']} unique labels")

    labels = load_label_files(label_root_path.joinpath(label_path_1))

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