import numpy as np

from Competitors import NoGKernel
from GraphData.DataSplits.load_splits import Load_Splits
from GraphData import GraphData


def main(db_name):
    # load the graph data
    graph_data = GraphData.GraphData()
    # set the path to the data
    data_path = "../../GraphData/DS_all/"
    graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                  relabel_nodes=True, use_features=False, use_attributes=False,
                                  distances_path=False)

    for validation_id in range(10):
        # three runs
        for i in range(3):
            data = Load_Splits("../GraphData/DataSplits", db_name)
            test_data = np.asarray(data[0][validation_id], dtype=int)
            training_data = np.asarray(data[1][validation_id], dtype=int)
            validate_data = np.asarray(data[2][validation_id], dtype=int)

            noG = NoGKernel.NoGKernel(graph_data, training_data, validate_data, test_data, i, "Results")
            noG.Run()



if __name__ == "__main__":
    main('NCI1')