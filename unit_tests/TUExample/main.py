from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset

from src.Experiment.ExperimentMain import ExperimentMain
from src.Preprocessing.GraphData.GraphData import ShareGNNDataset
from src.Preprocessing.load_labels import load_labels


def test_label_loading(graph_data: ShareGNNDataset):
    """
    Test loading of different types of labels:
    - Primary labels
    - WL labels with different depths
    - WL labeled labels with different depths
    """
    experiment = ExperimentMain(Path('unit_tests/TUExample/configs/config_main.yml'))
    experiment.ExperimentPreprocessing()

    # store all labels in a dictionary for easy access and comparison
    all_labels = {}

    # Get the dataset name and label path
    dataset_name = "MUTAG"
    label_path = Path('unit_tests/TUExample/Data/Labels').joinpath(dataset_name)

    # Test loading primary labels
    primary_label_path = label_path.joinpath(f"{dataset_name}_labels_primary.pt")
    primary_labels = load_labels(primary_label_path)
    print(f"Primary labels loaded successfully:")
    print(f"  Dataset name: {primary_labels.dataset_name}")
    print(f"  Label name: {primary_labels.label_name}")
    print(f"  Number of unique labels: {primary_labels.num_unique_node_labels}")
    all_labels['primary'] = primary_labels

    # Test loading WL labels with different depths
    for depth in [0, 1, 2, 3]:
        wl_label_path = label_path.joinpath(f"{dataset_name}_labels_wl_{depth}.pt")
        wl_labels = load_labels(wl_label_path)
        print(f"WL labels (depth {depth}) loaded successfully:")
        print(f"  Dataset name: {wl_labels.dataset_name}")
        print(f"  Label name: {wl_labels.label_name}")
        print(f"  Number of unique labels: {wl_labels.num_unique_node_labels}")
        all_labels[f'wl_depth_{depth}'] = wl_labels

    # Test loading WL labeled labels with different depths
    for depth in [0, 1, 2, 3]:
        wl_labeled_path = label_path.joinpath(f"{dataset_name}_labels_wl_labeled_{depth}.pt")
        wl_labeled_labels = load_labels(wl_labeled_path)
        print(f"WL labeled labels (depth {depth}) loaded successfully:")
        print(f"  Dataset name: {wl_labeled_labels.dataset_name}")
        print(f"  Label name: {wl_labeled_labels.label_name}")
        print(f"  Number of unique labels: {wl_labeled_labels.num_unique_node_labels}")
        all_labels[f'wl_labeled_depth_{depth}'] = wl_labeled_labels

    # add wl_edge_labels
    for depth in [0, 1, 2, 3]:
        wl_edge_label_path = label_path.joinpath(f"{dataset_name}_labels_wl_labeled_edges_{depth}.pt")
        wl_edge_labels = load_labels(wl_edge_label_path)
        print(f"WL edge labels (depth {depth}) loaded successfully:")
        print(f"  Dataset name: {wl_edge_labels.dataset_name}")
        print(f"  Label name: {wl_edge_labels.label_name}")
        print(f"  Number of unique labels: {wl_edge_labels.num_unique_node_labels}")
        all_labels[f'wl_edge_depth_{depth}'] = wl_edge_labels

    # check whether wl_1 and wl_labeled_1 are not equal
    assert all_labels['wl_depth_1'].num_unique_node_labels < all_labels['wl_labeled_depth_1'].num_unique_node_labels, \
        "WL labels with depth 1 should have fewer unique labels than WL labeled labels with depth 1."
    # check wheterh primary and wl_1 are not equal
    assert all_labels['primary'].num_unique_node_labels < all_labels['wl_depth_1'].num_unique_node_labels, \
        "Primary labels should have fewer unique labels than WL labels with depth 1."
    # check whether wl_2 and wl_labeled_2 are not equal
    assert all_labels['wl_depth_2'].num_unique_node_labels < all_labels['wl_labeled_depth_2'].num_unique_node_labels, \
        "WL labels with depth 2 should have fewer unique labels than WL labeled labels with depth 2."
    # check whether wl_1 and wl_2 are not equal
    assert all_labels['wl_depth_1'].num_unique_node_labels < all_labels['wl_depth_2'].num_unique_node_labels, \
        "WL labels with depth 1 should have fewer unique labels than WL labels with depth 2."



    # load the original data
    original_graph_data = TUDataset(root='tmp/', name=graph_data.name, use_node_attr=True, use_edge_attr=True)

    # plot the first graph
    first_graph = graph_data.nx_graphs[0]
    # plot the first graph
    import matplotlib.pyplot as plt
    import networkx as nx
    plt.figure(figsize=(8, 6))
    # use kawai for pos
    pos = nx.kamada_kawai_layout(first_graph)
    nx.draw(first_graph, pos=pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title(f"Graph: {graph_data.name} - First Graph")
    plt.show()

    # check if the labels of the first graph are correct
    first_graph_labels = {}
    for key, value in all_labels.items():
        first_graph_labels[key] = value.node_labels[graph_data.slices['primary_node_labels'][0]:graph_data.slices['primary_node_labels'][1]]

    # for primary the first 14 have to be the same label
    assert all(first_graph_labels['primary'][:14] == 0), "The first 14 nodes should have the primary label 0."
    assert all(first_graph_labels['primary'][14:15] == 2), "The 15th node should have the primary label 2."
    assert all(first_graph_labels['primary'][15:] == 1), "All nodes after the 15th should have the primary label 1."
    # check degree labels
    assert len(torch.unique(first_graph_labels['wl_depth_0'][[0,1,2,5,6,7,10,11,13]])) == 1, "All those nodes have degree 2."
    assert len(torch.unique(first_graph_labels['wl_depth_0'][[3,4,8,9,12,14]])) == 1, "All those nodes have degree 3."
    assert len(torch.unique(first_graph_labels['wl_depth_0'][[15,16]])) == 1, "All those nodes have degree 1."
    assert len(torch.unique(first_graph_labels['wl_depth_0'][[0,3,15]])) == 3, "All those nodes have different degrees."
    # check labeled degree labels
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[0,1,2,5,6,7,10,11,13]])) == 1, "All those nodes have the same labeled degree."
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[3,4,8,9]])) == 1, "All those nodes have the same labeled degree."
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[12]])) == 1, "All those nodes have the same labeled degree."
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[14]])) == 1, "All those nodes have the same labeled degree."
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[15,16]])) == 1, "All those nodes have the same labeled degree."
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_0'][[0,3,12,14,15]])) == 5, "All those nodes have different labeled degrees."
    # check wl_1 labels
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[0,1]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[15,16]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[2,5,6,7, 10, 11]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[3,9]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[4,8,12]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[13]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[14]])) == 1, "All those nodes have the same WL label at depth 1."
    assert len(torch.unique(first_graph_labels['wl_depth_1'][[0,2,3,4,13,14,15]])) == 7, "All those nodes have different WL labels at depth 1."
    # check wl_1 labeled labels
    assert len(torch.unique(first_graph_labels['wl_labeled_depth_1'][[0,1]])) == 1, "All those nodes have the same WL labeled label at depth 1."

    pass


def main():
    experiment = ExperimentMain(Path('unit_tests/TUExample/configs/config_main.yml'))
    experiment.ExperimentPreprocessing()
    # load the preprocessed data
    mutag_data = ShareGNNDataset(
        root=Path('unit_tests/TUExample/Data/Graphs'),
        name='MUTAG',
        task='graph_classification',
    )
    # create nx_graphs
    mutag_data.create_nx_graphs()

    # Test label loading
    test_label_loading(mutag_data)

    # Continue with the original test
    experiment.run_configurations()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)

if __name__ == '__main__':
    main()
