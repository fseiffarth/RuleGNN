import json

from GraphData import GraphData
from TrainTestData import TrainTestData as ttd


def create_splits(db_name, path="../../../GraphData/DS_all/"):
    splits = []
    run_id = 1
    k = 10
    graph_data = GraphData.GraphData()
    graph_data.init_from_graph_db(path, db_name, with_distances=False, with_cycles=False,
                                  relabel_nodes=True, use_features=False, use_attributes=False,
                                  distances_path=None)

    run_test_indices = ttd.get_data_indices(graph_data.num_graphs, seed=run_id, kFold=k)
    for validation_id in range(0, k):
        seed = validation_id + k * run_id

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


if __name__ == "__main__":
    create_splits("MUTAG")
    #create_splits("DHFR")
    #create_splits("Mutagenicity")
    #create_splits("NCI109")
    #create_splits("SYNTHETICnew")
