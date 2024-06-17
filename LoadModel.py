import os

import click
import numpy as np
import torch
import yaml

from GraphData.DataSplits.load_splits import Load_Splits
from GraphData.GraphData import get_graph_data
from GraphData.Labels.generator.load_labels import load_labels
from Layers.GraphLayers import Layer
from NeuralNetArchitectures import GraphNN
from utils.Parameters.Parameters import Parameters
from utils.RunConfiguration import RunConfiguration


@click.command()
@click.option('--db', default="EvenOddRings2_16", help='Database to use')
@click.option('--config', default="", help='Path to the configuration file')
# --data_path ../GraphBenchmarks/Data/ --db EvenOddRings2_16 --config ../TEMP/EvenOddRings2_16/config.yml
def main(db, config):
    run = 0
    k_val = 0
    kFold = 10
    # load the model
    configs = yaml.safe_load(open(config))
    # get the data path from the config file
    data_path = f"{configs['paths']['data']}"
    r_path = f"{configs['paths']['results']}"
    distance_path = f"{configs['paths']['distances']}"
    splits_path = f"{configs['paths']['splits']}"
    results_path = r_path + db + "/Results/"

    graph_data = get_graph_data(data_path=data_path, db_name=db, distance_path=distance_path,
                                use_features=configs['use_features'], use_attributes=configs['use_attributes'])
    # adapt the precision of the input data
    if 'precision' in configs:
        if configs['precision'] == 'double':
            for i in range(len(graph_data.inputs)):
                graph_data.inputs[i] = graph_data.inputs[i].double()

    #create run config from first config
    # get all different run configurations
    # define the network type from the config file
    run_configs = []
    # iterate over all network architectures
    for network_architecture in configs['networks']:
        layers = []
        # get all different run configurations
        for l in network_architecture:
            layers.append(Layer(l))
        for b in configs['batch_size']:
            for lr in configs['learning_rate']:
                for e in configs['epochs']:
                    for d in configs['dropout']:
                        for o in configs['optimizer']:
                            for loss in configs['loss']:
                                run_configs.append(RunConfiguration(network_architecture, layers, b, lr, e, d, o, loss))
    for i, run_config in enumerate(run_configs):
        config_id = str(i).zfill(6)
        model_path = f'{r_path}{db}/Models/model_Configuration_{config_id}_run_{run}_val_step_{k_val}.pt'
        seed = k_val + kFold * run
        data = Load_Splits(f"{configs['paths']['splits']}", db)
        test_data = np.asarray(data[0][k_val], dtype=int)
        training_data = np.asarray(data[1][k_val], dtype=int)
        validate_data = np.asarray(data[2][k_val], dtype=int)
        # check if the model exists
        try:
            with open(model_path, 'r'):
                """
                Set up the network and the parameters
                """

                para = Parameters()
                """
                    Data parameters
                """
                para.set_data_param(path=data_path, results_path=results_path,
                                    splits_path=splits_path,
                                    db=db,
                                    max_coding=1,
                                    layers=run_config.layers,
                                    batch_size=run_config.batch_size, node_features=1,
                                    load_splits=configs['load_splits'],
                                    configs=configs,
                                    run_config=run_config, )

                """
                    Network parameters
                """
                para.set_evaluation_param(run_id=run, n_val_runs=kFold, validation_id=k_val,
                                          config_id=config_id,
                                          n_epochs=run_config.epochs,
                                          learning_rate=run_config.lr, dropout=run_config.dropout,
                                          balance_data=configs['balance_training'],
                                          convolution_grad=True,
                                          resize_graph=True)

                """
                Print, save and draw parameters
                """
                para.set_print_param(no_print=False, print_results=False, net_print_weights=True,
                                     print_number=1,
                                     draw=False, save_weights=False,
                                     save_prediction_values=False, plot_graphs=False,
                                     print_layer_init=False)

                for l in run_config.layers:
                    label_path = f"GraphData/Labels/{db}_{l.get_layer_string()}_labels.txt"
                    if os.path.exists(label_path):
                        g_labels = load_labels(path=label_path)
                        graph_data.node_labels[l.get_layer_string()] = g_labels
                    else:
                        # raise an error if the file does not exist
                        raise FileNotFoundError(f"File {label_path} does not exist")

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)

                net = GraphNN.GraphNet(graph_data=graph_data,
                                       para=para,
                                       seed=seed)

                net.load_state_dict(torch.load(model_path))
                # iterate over all layers and get the number of zero resp. non-zero parameters
                for layer in net.net_layers:
                    print(f"Layer {layer.name}")
                    print(f"Number of non-zero parameters: {torch.count_nonzero(layer.Param_W)}")
                    print(f"Number of zero parameters: {torch.numel(layer.Param_W) - torch.count_nonzero(layer.Param_W)}")
                    # print non-zero parameters in percent
                    print(f"Percentage of non-zero parameters: {torch.count_nonzero(layer.Param_W) / torch.numel(layer.Param_W) * 100}%")
                # evaluate the performance of the model on the test data
                outputs = torch.zeros((len(test_data), graph_data.num_classes), dtype=torch.double)
                with torch.no_grad():
                    for j, data_pos in enumerate(test_data, 0):
                        inputs = torch.DoubleTensor(graph_data.inputs[data_pos])
                        outputs[j] = net(inputs, data_pos)
                    labels = graph_data.one_hot_labels[test_data]
                    # calculate the errors between the outputs and the labels by getting the argmax of the outputs and the labels
                    counter = 0
                    correct = 0
                    for i, x in enumerate(outputs, 0):
                        if torch.argmax(x) == torch.argmax(labels[i]):
                            correct += 1
                        counter += 1
                    accuracy = correct / counter
                    print(f"Accuracy for model {model_path} is {accuracy}")
        except FileNotFoundError:
            print(f"Model {model_path} not found")
            return

if __name__ == '__main__':
    main()