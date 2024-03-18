# Evaluate the performance of the trained models on the test data
import torch

import TrainTestData.TrainTestData as ttd
from GraphData import GraphData
from NeuralNetArchitectures import GraphNN

# create click argument parser
import click


@click.command()
@click.option('--data_path', default="../../GraphData/DS_all/", help='Path to the graph data')
@click.option('--db', default="MUTAG", help='Database to use')
@click.option('--node_features', default=1, help='Number of node features')
@click.option('--node_labels', default=7, help='Number of node labels')
@click.option('--edge_labels', default=4, help='Number of edge labels')
@click.option('--num_classes', default=2, help='Number of classes')
def main(data_path, db, node_features, node_labels, edge_labels, num_classes):
    path = data_path
    distance_list = []
    cycle_list = []

    graph_data = GraphData.GraphData()
    graph_data.init_from_graph_db(data_path, db, with_distances=False, with_cycles=False)

    kFold = 10
    runs = 10

    accuracies = []

    for run in range(runs):
        run_test_indices = ttd.get_data_indices(graph_data.num_graphs, seed=run, kFold=kFold)
        for k_val in range(kFold):
            model_path = f'../Results/{db}/Models/model_run_{run}_val_step_{k_val}.pt'
            seed = k_val + kFold * run
            # check if the model exists
            try:
                with open(model_path, 'r'):
                    training_data, validate_data, test_data = ttd.get_train_validation_test_list(test_indices=run_test_indices,
                                                                                     validation_step=k_val,
                                                                                     seed=seed,
                                                                                     balanced=True,
                                                                                     graph_labels=graph_data.graph_labels,
                                                                                     val_size=0.1)

                    # load the model and evaluate the performance on the test data
                    net = GraphNN.GraphNetOriginal(graph_data=graph_data, n_node_features=node_features,
                                                   n_node_labels=node_labels, n_edge_labels=edge_labels,
                                                   seed=seed,
                                                   dropout=0,
                                                   out_classes=num_classes,
                                                   print_weights=False)
                    net.load_state_dict(torch.load(model_path))
                    # evaluate the performance of the model on the test data
                    outputs = torch.zeros((len(test_data), num_classes), dtype=torch.double)
                    with torch.no_grad():
                        for j, data_pos in enumerate(test_data, 0):
                            inputs = torch.DoubleTensor(graph_data.inputs[data_pos])
                            outputs[j] = net(inputs, data_pos)
                        labels = graph_data.one_hot_labels[test_data]
                        test_acc = 100 * ttd.get_accuracy(outputs, labels, one_hot_encoding=True)
                        accuracies.append(test_acc)
                        print(f"Run {run} Validation Step: {k_val} Test Accuracy: {test_acc}")

            except FileNotFoundError:
                print(f"Model {model_path} not found")
                continue
    print(accuracies)
    print(f"Average accuracy: {sum(accuracies) / len(accuracies)}")


if __name__ == "__main__":
    main()
