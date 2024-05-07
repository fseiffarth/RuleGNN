import numpy as np

from GraphData.GraphData import GraphData, get_graph_data
from Kernels.NoGKernel import NoGKernel
from Kernels.GraphKernels import WLKernel
from GraphData.DataSplits.load_splits import Load_Splits
from LoadData.csl import CSL


def main(db_name, data_path="../../GraphData/DS_all/"):
    #datapath = "/home/mlai21/seiffart/Data/GraphData/DS_all/"
    # load the graph data
    graph_data = get_graph_data(db_name, data_path=data_path, distance_path="../GraphData/Distances/")

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

            wlKernel = WLKernel(graph_data, run_num=i, validation_num=validation_id,
                                training_data=training_data, validate_data=validate_data, test_data=test_data,
                                seed=i)
            wlKernel.Run()

            noG = NoGKernel(graph_data, run_num=i, validation_num=validation_id, training_data=training_data,
                            validate_data=validate_data, test_data=test_data, seed=i)
            noG.Run()



if __name__ == "__main__":
    # run parallel for all datasets
    # joblib.Parallel(n_jobs=-1)(
    #     joblib.delayed(main)(db_name) for db_name in ['CSL', 'DHFR', 'SYNTHETICnew', 'NCI1', 'NCI109', 'Mutagenicity'])
    #main("IMDB-MULTI")
    #main("SnowflakesCount", data_path="../GraphBenchmarks/Data/")
    main("LongRings100", data_path="../GraphBenchmarks/Data/")
