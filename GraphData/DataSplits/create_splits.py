import json

from GraphData import GraphData
from GraphData.GraphData import get_graph_data
from TrainTestData import TrainTestData as ttd

def zinc_splits():
    splits = []
    db_name = "ZINC"
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


def create_splits(db_name, path="../../../GraphData/DS_all/", output_path=""):
    splits = []
    run_id = 0
    k = 10
    graph_data = get_graph_data(db_name, path)

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
        with open(f"{output_path}{db_name}_train.txt", "a") as f:
            f.write(" ".join([str(x) for x in training_data]))
            f.write("\n")
        with open(f"{output_path}{db_name}_validation.txt", "a") as f:
            f.write(" ".join([str(x) for x in validate_data]))
            f.write("\n")
        with open(f"{output_path}{db_name}_test.txt", "a") as f:
            f.write(" ".join([str(x) for x in test_data]))
            f.write("\n")

        splits.append({"test": test_data, "model_selection": [{"train": training_data, "validation": validate_data}]})

    # save splits to json as one line use json.dumps
    with open(f"{output_path}{db_name}_splits.json", "w") as f:
        f.write(json.dumps(splits))


if __name__ == "__main__":

    zinc_splits()
    #create_splits("DHFR")
    #create_splits("Mutagenicity")
    #create_splits("NCI109")
    #create_splits("SYNTHETICnew")
    # for db in ["DHFR", "Mutagenicity", "NCI109", "SYNTHETICnew", "MUTAG"]:
    #     create_splits(db)
