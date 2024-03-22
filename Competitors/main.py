import click
import joblib
import numpy as np

from Kernels.NoGKernel import NoGKernel
from Kernels.WLKernel import WLKernel
from GraphData.DataSplits.load_splits import Load_Splits
from GraphData.GraphData import GraphData
from LoadData.csl import CSL


# click argument for data path
@click.option("--data_path", default="../../GraphData/DS_all/", help="Path to the data")
def main(data_path="../../GraphData/DS_all/"):
    # run parallel for all datasets
    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(run_competitors)(db_name, data_path) for db_name in ['CSL', 'DHFR', 'SYNTHETICnew', 'NCI1', 'NCI109', 'Mutagenicity'])


def run_competitors(db_name, data_path="../../GraphData/DS_all/"):
    # load the graph data
    graph_data = GraphData()
    if db_name == "CSL":
        csl = CSL()
        graph_data = csl.get_graphs(with_distances=False)
    else:
        graph_data.init_from_graph_db(data_path, db_name, with_distances=False, with_cycles=False,
                                      relabel_nodes=True, use_features=False, use_attributes=False,
                                      distances_path=False)

    validation_size = 10
    if db_name == "CSL":
        validation_size = 5
    for validation_id in range(validation_size):
        # three runs
        for i in range(3):
            data = Load_Splits("../GraphData/DataSplits", db_name)
            test_data = np.asarray(data[0][validation_id], dtype=int)
            training_data = np.asarray(data[1][validation_id], dtype=int)
            validate_data = np.asarray(data[2][validation_id], dtype=int)

            noG = NoGKernel(graph_data, run_num=i, validation_num=validation_id, training_data=training_data,
                                      validate_data=validate_data, test_data=test_data, seed=i)
            noG.Run()
            wlKernel = WLKernel(graph_data, run_num=i, validation_num=validation_id,
                                         training_data=training_data, validate_data=validate_data, test_data=test_data,
                                         seed=i)
            wlKernel.Run()


if __name__ == "__main__":
    main()
