'''
Created on 15.03.2019

@author: florian
'''
import os

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from joblib import Parallel, delayed

from DataSplits.load_splits import Load_Splits
from GraphData.Labels.load_labels import load_labels
from Layers.GraphLayers import Layer
from LoadData.csl import PrepareCSL
from Methods.NoGKernelWLNN import NoGKernelWLNN
from TrainTestData import TrainTestData as ttd
from GraphData import GraphData, NodeLabeling, EdgeLabeling
from Methods.GraphRuleMethod import GraphRuleMethod
from Methods.NoGKernel import NoGKernel
from Methods.NoGKernelNN import NoGKernelNN
from Parameters import Parameters
import ReadWriteGraphs.GraphDataToGraphList as gdtgl


@click.command()
@click.option('--data_path', default="../GraphData/DS_all/", type=str, help='Path to the graph data')
@click.option('--distances_path', default=None, type=str, help='Path to the distances')
@click.option('--graph_db_name', default="MUTAG", type=str, help='Database name')
@click.option('--network_type', default='wl_1:1,2,3;wl_1', type=str, help='Layer types of the network')
@click.option('--batch_size', default=16, type=int)
@click.option('--edge_labels', default=-1, type=int,
              help='If None, the number of edge labels will be determined by the graph data')
@click.option('--use_features', default=True, type=bool,
              help='If true, the features of the nodes will be used, otherwise ones are used as features for every node')
@click.option('--use_attributes', default=False, type=bool,
              help='If true, the attributes of the nodes will be used instead of the labels')
@click.option('--load_splits', default=False, type=bool, help='If true, the splits will be loaded from the file')
@click.option('--convolution_grad', default=True, type=bool, help='If true, the convolutional layer will be trained')
@click.option('--resize_grad', default=True, type=bool, help='If true, the resize layer will be trained')
@click.option('--max_coding', default=1, type=int)
@click.option('--node_features', default=1, type=int)
@click.option('--run_id', default=0, type=int)
@click.option('--validation_number', default=10, type=int)
@click.option('--validation_id', default=0, type=int)
@click.option('--epochs', default=100, type=int)
@click.option('--lr', default=0.01, type=float)
@click.option('--dropout', default=0.0, type=float)
# balanced data option
@click.option('--balanced', default=False, type=bool)
# turn off all drawing and printing options
@click.option('--no_print', default=False, type=bool)
# drawing option
@click.option('--draw', default=False, type=bool)
# printing and saving weights option
@click.option('--save_weights', default=False, type=bool)
@click.option('--save_prediction_values', default=False, type=bool)
@click.option('--print_results', default=False, type=bool)
# plot the graphs
@click.option('--plot_graphs', default=False, type=bool)
# debug vs fast mode option in click vs custom option
@click.option('--mode', default="normal", type=click.Choice(['fast', 'debug', 'normal']))
# some example parameters
# --db PTC_FM --epochs 200 --batch_size 32 --node_labels 18 --edge_labels 4 --draw True
# --db MUTAG --epochs 200 --batch_size 32 --node_labels 7 --edge_labels 4 --draw True

# current configuration
# --db MUTAG --epochs 500 --batch_size 64 --node_labels 7 --edge_labels 4 --lr 0.001 --balanced True --draw True --save_weights True --save_prediction_values False

# --data_path ../GraphData/DS_all/ --graph_db_name MUTAG --threads 10 --runs 10 --epochs 1000 --batch_size 64 --node_labels 7 --edge_labels 4 --lr 0.005 --balanced True --mode debug

# --data_path ../GraphData/DS_all/ --graph_db_name MUTAG --threads 10 --runs 10 --epochs 1000 --batch_size 64 --node_labels 7 --edge_labels 4 --lr 0.005 --balanced True --mode debug

def main(data_path, distances_path, graph_db_name, max_coding, network_type, batch_size, node_features, edge_labels, run_id,
         validation_number, validation_id,
         epochs, lr, dropout, balanced, no_print, draw, save_weights, save_prediction_values, print_results,
         plot_graphs, mode, convolution_grad, resize_grad, use_features, use_attributes, load_splits):
    # if not exists create the results directory
    if not os.path.exists("Results"):
        try:
            os.makedirs("Results")
        except:
            pass
    # if not exists create the directory for db under Results
    if not os.path.exists("Results/" + graph_db_name):
        try:
            os.makedirs("Results/" + graph_db_name)
        except:
            pass
    # if not exists create the directory Results, Plots, Weights and Models under db
    if not os.path.exists("Results/" + graph_db_name + "/Results"):
        try:
            os.makedirs("Results/" + graph_db_name + "/Results")
        except:
            pass
    if not os.path.exists("Results/" + graph_db_name + "/Plots"):
        try:
            os.makedirs("Results/" + graph_db_name + "/Plots")
        except:
            pass
    if not os.path.exists("Results/" + graph_db_name + "/Weights"):
        try:
            os.makedirs("Results/" + graph_db_name + "/Weights")
        except:
            pass
    if not os.path.exists("Results/" + graph_db_name + "/Models"):
        try:
            os.makedirs("Results/" + graph_db_name + "/Models")
        except:
            pass

    plt.ion()

    # path do db and db
    results_path = "Results/" + graph_db_name + "/Results/"
    print_layer_init = True
    # if debug mode is on, turn on all print and draw options
    if mode == "debug":
        draw = True
        print_results = True
        save_prediction_values = True
        threads = 1
    # if fast mode is on, turn off all print and draw options
    if mode == "fast":
        draw = False
        print_results = False
        save_weights = False
        save_prediction_values = False
        plot_graphs = False
        print_layer_init = False

    """
    Create Input data, information and labels from the graphs for training and testing
    """
    if graph_db_name == "CSL":
        csl = PrepareCSL(root="LoadData/Datasets/CSL/")
        graph_data = csl.graph_data(with_distances=True, with_cycles=True)
        # TODO: find other labeling method
    else:
        graph_data = GraphData.GraphData()
        graph_data.init_from_graph_db(data_path, graph_db_name, with_distances=True, with_cycles=False,
                                      relabel_nodes=True, use_features=use_features, use_attributes=use_attributes, distances_path=distances_path)

    # split layer_types into a list
    network_type = network_type.split(";")
    # first entry is the max node label
    node_labels = int(network_type[0])
    layers = []
    for i,l in enumerate(network_type[1:], 1):
        try:
            int(l)
        except:
            # if network_type[i-1] is an integer, add the layer
            try:
                num_labels = int(network_type[i-1])
                layer = Layer(l, num_labels)
                layers.append(layer)
            except:
                layer = Layer(l, node_labels)
                layers.append(layer)

    # add node labels for each layer_name except for the primary
    for l in layers:
        if l.rule_name != "primary":
            # if wl is in the layer name, then it is a weisfeiler lehman layer
            iterations = 0
            if "wl" in l.rule_name:
                if "max" in l.rule_name:
                    # set the max iterations to 20
                    iterations = 20
                else:
                    try:
                        # split by _ and get the number of iterations
                        iterations = int(l.rule_name.split("_")[1])
                    except:
                        pass
                if iterations == 0:
                    l.node_labels = -1
                    l_string = ""
                    max_label_num = None
                else:
                    l_string = f'_{l.node_labels}'
                    max_label_num = l.node_labels
                if os.path.exists(f"GraphData/Labels/{graph_db_name}_{l.rule_name}{l_string}_labels.txt"):
                    g_labels = load_labels(db_name=graph_db_name,label_type=l.rule_name, max_label_num=max_label_num,path=f"GraphData/Labels/")
                    graph_data.node_labels[l.rule_name] = g_labels
                else:
                    if iterations == 0:
                        graph_data.add_node_labels(node_labeling_name=l.rule_name, max_label_num=l.node_labels,
                                                   node_labeling_method=NodeLabeling.degree_node_labeling)
                    else:
                        graph_data.add_node_labels(node_labeling_name=l.rule_name, max_label_num=l.node_labels,
                                                   node_labeling_method=NodeLabeling.weisfeiler_lehman_node_labeling,
                                                   max_iterations=iterations)

    para = Parameters.Parameters()

    """
        Data parameters
    """
    para.set_data_param(path=data_path, results_path=results_path,
                        db=graph_db_name,
                        max_coding=max_coding,
                        network_type=network_type,
                        layers=layers,
                        batch_size=batch_size, node_features=node_features,
                        load_splits=load_splits)

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_number, validation_id=validation_id, n_epochs=epochs,
                              learning_rate=lr,
                              dropout=dropout,
                              balance_data=balanced,
                              convolution_grad=convolution_grad,
                              resize_graph=resize_grad)

    """
    Print, save and draw parameters
    """
    para.set_print_param(no_print=no_print, print_results=print_results, net_print_weights=True, print_number=1,
                         draw=draw, save_weights=save_weights,
                         save_prediction_values=save_prediction_values, plot_graphs=plot_graphs,
                         print_layer_init=print_layer_init)

    """
        Get the first index in the results directory that is not used
    """
    para.set_file_index(size=6)

    if para.plot_graphs:
        # if not exists create the directory
        if not os.path.exists(f"Results/{para.db}/Plots"):
            os.makedirs(f"Results/{para.db}/Plots")
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             f"Results/{para.db}/Plots/graph_{str(i).zfill(5)}.png")

    validation_step(para.run_id, para.validation_id, graph_data, para, results_path)


def validation_step(run_id, validation_id, graph_data: GraphData.GraphData, para: Parameters.Parameters, results_path):
    """
    Split the data in training validation and test set
    """
    seed = validation_id + para.n_val_runs * run_id
    if para.load_splits is False:
        run_test_indices = ttd.get_data_indices(graph_data.num_graphs, seed=run_id, kFold=para.n_val_runs)
        """
        Create the data
        """
        training_data, validate_data, test_data = ttd.get_train_validation_test_list(test_indices=run_test_indices,
                                                                                     validation_step=validation_id,
                                                                                     seed=seed,
                                                                                     balanced=para.balance_data,
                                                                                     graph_labels=graph_data.graph_labels,
                                                                                     val_size=0.1)
    else:
        data = Load_Splits("DataSplits", para.db)
        test_data = np.asarray(data[0][validation_id], dtype=int)
        training_data = np.asarray(data[1][validation_id], dtype=int)
        validate_data = np.asarray(data[2][validation_id], dtype=int)

    # print train resp test data to some file
    # with open(f"{results_path}train_test_indices.txt", "a") as f:
    #     f.write(f"Run: {run}\n")
    #     f.write(f"Validation step: {k_val}\n")
    #     # separate the indices by a space
    #     training_data = " ".join(map(str, training_data))
    #     f.write(f"Train indices:\n{training_data}\n")
    #     test_data = " ".join(map(str, test_data))
    #     f.write(f"Test indices:\n{test_data}\n")

    method = GraphRuleMethod(run_id, validation_id, graph_data, training_data, validate_data, test_data, seed, para,
                             results_path)
    # method = NoGKernel(run, k_val, graph_data, training_data, validate_data, test_data, seed, para, results_path)
    # method = NoGKernelNN(run, k_val, graph_data, training_data, validate_data, test_data, seed, para, results_path)
    # method = NoGKernelWLNN(run, k_val, graph_data, training_data, validate_data, test_data, seed, para, results_path)

    """
    Run the method
    """
    method.Run()


if __name__ == '__main__':
    main()
