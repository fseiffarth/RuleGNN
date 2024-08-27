import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.utils.GraphData import get_graph_data
from src.TrainTestData import TrainTestData as ttd


def zinc_splits():
    splits = []
    db_name = "ZINC_original"
    # Dict use double quotes
    training_data = [i for i in range(0, 10000)]
    validate_data = [i for i in range(10000, 11000)]
    test_data = [i for i in range(11000, 12000)]
    # write data to txt file
    with open(f"{db_name}_train.txt", "a") as f:
        f.write(" ".join([str(x) for x in training_data]))
        f.write("\n")
    with open(f"{db_name}_validation.txt", "a") as f:
        f.write(" ".join([str(x) for x in validate_data]))
        f.write("\n")
    with open(f"{db_name}_test.txt", "a") as f:
        f.write(" ".join([str(x) for x in test_data]))
        f.write("\n")

    splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

    # save splits to json as one line use json.dumps
    with open(f"{db_name}_splits.json", "w") as f:
        f.write(json.dumps(splits))


def create_transfer_splits(db_name, path="../GraphData/DS_all/", output_path="Data/Splits/", data_format=None, split_type='random'):
    '''
    Create splits for transfer learning
    :param db_name: name of the dataset
    :param path: path to the dataset
    :param output_path: path to save the splits
    :param data_format: format of the dataset
    :param split_type: type of the split, transfer takes all but the last graph dataset as training and the last as test, mixed takes training and testing equally from all graph datasets, random takes random graphs from all datasets
    '''

    # string to int
    int_type = 0
    for i in range(len(split_type)):
        int_type += ord(split_type[i])
    seed = 687384987 + int_type
    np.random.seed(seed)
    if split_type == "random":
        create_splits(db_name, path, output_path, data_format)
        return
    k = 10
    graph_data = get_graph_data(db_name, path, data_format=data_format)
    # number of graphs per graph_dataset
    graph_number_map = defaultdict(list)
    graph_datasets = []
    for i, graph in enumerate(graph_data.graphs):
        graph_number_map[graph.name].append(i)
        if graph.name not in graph_datasets:
            graph_datasets.append(graph.name)

    if split_type == "transfer":
        test_indices = graph_number_map[graph_datasets[-1]]
        # duplicate test_data k times
        test_data = [test_indices for i in range(k)]
        # shuffle the training data and create k folds
        training_indices = []
        for dataset in graph_datasets[:-1]:
            training_indices += graph_number_map[dataset]
        training_indices = np.random.permutation(training_indices)
        # split the training data into k folds
        data_splits = np.array_split(training_indices, k)

        splits = []
        for i in range(k):
            training_data = []
            validate_data = []
            # append all the training data except the validation data
            training_data += [data for j, data in enumerate(data_splits) if j != i]
            # flatten the list and sort it
            training_data = np.sort(np.concatenate(training_data)).tolist()
            # append the validation data
            # sort the validation data
            validate_data = np.sort(data_splits[i]).tolist()
            splits.append(
                {"test": test_data[i], "model_selection": [{"train": training_data, "validation": validate_data}]})

        # save splits to json as one line use json.dumps
        with open(f"{output_path}{db_name}_splits_transfer.json", "w") as f:
            f.write(json.dumps(splits))
    elif split_type == "mixed":
        # split the graph_datasets equally into k parts
        parts = []
        for i in range(len(graph_datasets)):
            shuffled_indices = np.random.permutation(graph_number_map[graph_datasets[i]])
            splits = np.array_split(shuffled_indices, k)
            # sort the splits
            for i, split in enumerate(splits):
                splits[i] = np.sort(split)
            parts.append(splits)
        run_test_indices = []
        for i in range(k):
            run_test_indices.append([])
            for j in parts:
                run_test_indices[i] += j[i].tolist()

        splits = []
        for validation_id in range(0, k):
            seed = 687384987 + validation_id + k

            """
            Create the data
            """
            training_data, validate_data, test_data = ttd.get_train_validation_test_list(test_indices=run_test_indices,
                                                                                         validation_step=validation_id,
                                                                                         seed=seed,
                                                                                         balanced=False,
                                                                                         graph_labels=graph_data.graph_labels,
                                                                                         val_size=1.0 / k)

            # Dict use double quotes
            training_data = [int(x) for x in training_data]
            validate_data = [int(x) for x in validate_data]
            test_data = [int(x) for x in test_data]
            # write data to txt file
            # with open(f"{output_path}{db_name}_train.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in training_data]))
            #    f.write("\n")
            # with open(f"{output_path}{db_name}_validation.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in validate_data]))
            #    f.write("\n")
            # with open(f"{output_path}{db_name}_test.txt", "a") as f:
            #    f.write(" ".join([str(x) for x in test_data]))
            #    f.write("\n")

            splits.append(
                {"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

        # save splits to json as one line use json.dumps
        with open(f"{output_path}{db_name}_splits_mixed.json", "w") as f:
            f.write(json.dumps(splits))


def create_splits(db_name: str, path: Path = Path("../GraphData/DS_all/"), output_path: Path=Path("Data/Splits/"), format=None):
    splits = []
    run_id = 0
    k = 10
    graph_data = get_graph_data(db_name, path, data_format=format)

    run_test_indices = ttd.get_data_indices(graph_data.num_graphs, seed=run_id, kFold=k)
    for validation_id in range(0, k):
        seed = 687384987 + validation_id + k * run_id

        """
        Create the data
        """
        training_data, validate_data, test_data = ttd.get_train_validation_test_list(test_indices=run_test_indices,
                                                                                     validation_step=validation_id,
                                                                                     seed=seed,
                                                                                     balanced=False,
                                                                                     graph_labels=graph_data.graph_labels,
                                                                                     val_size=1.0 / k)

        # Dict use double quotes
        training_data = [int(x) for x in training_data]
        validate_data = [int(x) for x in validate_data]
        test_data = [int(x) for x in test_data]
        # write data to txt file
        #with open(f"{output_path}{db_name}_train.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in training_data]))
        #    f.write("\n")
        #with open(f"{output_path}{db_name}_validation.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in validate_data]))
        #    f.write("\n")
        #with open(f"{output_path}{db_name}_test.txt", "a") as f:
        #    f.write(" ".join([str(x) for x in test_data]))
        #    f.write("\n")

        splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

    # save splits to json as one line use json.dumps
    with open(output_path.joinpath(f"{db_name}_splits.json"), "w") as f:
        f.write(json.dumps(splits))


if __name__ == "__main__":

    zinc_splits()
    #create_splits("DHFR")
    #create_splits("Mutagenicity")
    #create_splits("NCI109")
    #create_splits("SYNTHETICnew")
    # for db in ["DHFR", "Mutagenicity", "NCI109", "SYNTHETICnew", "MUTAG"]:
    #     create_splits(db)
    for db in ['PTC_FM', 'PTC_FR', 'PTC_MM', 'PTC_MR']:
        create_splits(db)
