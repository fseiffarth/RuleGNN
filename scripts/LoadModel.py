import os

import click
import numpy as np
import torch
import yaml

from ModelSelection import get_run_configs
from utils.GraphData import get_graph_data
from utils.load_labels import load_labels
from Preprocessing.create_labels import save_node_labels
from Architectures.RuleGNN.RuleGNNLayers import Layer
from Architectures import RuleGNN
from utils.GraphLabels import Properties, combine_node_labels
from utils.Parameters.Parameters import Parameters
from utils.RunConfiguration import RunConfiguration
from utils.load_splits import Load_Splits


@click.command()
@click.option('--data_path', default="../GraphData/DS_all/", help='Path to the data')
@click.option('--out_path', default="", help='Path to the results')
@click.option('--db', default="EvenOddRings2_16", help='Database to use')
# --db DHFR --data_path ../GraphData/DS_all/ --out_path Results_Paper_Reproduced/
def main(data_path, out_path, db):
    run = 0
    k_val = 0
    kFold = 10
    # load the model
    config_path = f"{out_path}{db}/config.yml"
    configs = yaml.safe_load(open(config_path))
    r_path = f"{configs['paths']['results']}"
    label_path = f"{configs['paths']['labels']}"
    properties_path = f"{configs['paths']['properties']}"
    splits_path = f"{configs['paths']['splits']}"
    results_path = r_path + db + "/Results/"

    graph_data = get_graph_data(data_path=data_path, db_name=db, use_features=configs['use_features'], use_attributes=configs['use_attributes'])
    # adapt the precision of the input data
    if 'precision' in configs:
        if configs['precision'] == 'double':
            for i in range(len(graph_data.inputs)):
                graph_data.inputs[i] = graph_data.inputs[i].double()

    # get the run configurations
    run_configs = get_run_configs(configs)

    for i, run_config in enumerate(run_configs):
        config_id = str(i).zfill(6)
        model_path = f'{out_path}{db}/Models/model_Configuration_{config_id}_run_{run}_val_step_{k_val}.pt'
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
                    BenchmarkGraphs parameters
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

                for i, l in enumerate(run_config.layers):
                    # add the labels to the graph data
                    label_path = f"{label_path}{db}_{l.get_layer_string()}_labels.txt"
                    if os.path.exists(label_path):
                        g_labels = load_labels(path=label_path)
                        graph_data.node_labels[l.get_layer_string()] = g_labels
                    elif l.layer_type == "combined":  # create combined file if it is a combined layer and the file does not exist
                        combined_labels = []
                        # get the labels for each layer in the combined layer
                        for x in l.layer_dict['sub_labels']:
                            sub_layer = Layer(x, i)
                            sub_label_path = f"{label_path}/{db}_{sub_layer.get_layer_string()}_labels.txt"
                            if os.path.exists(sub_label_path):
                                g_labels = load_labels(path=sub_label_path)
                                combined_labels.append(g_labels)
                            else:
                                # raise an error if the file does not exist
                                raise FileNotFoundError(f"File {sub_label_path} does not exist")
                        # combine the labels and save them
                        g_labels = combine_node_labels(combined_labels)
                        graph_data.node_labels[l.get_layer_string()] = g_labels
                        save_node_labels(data_path=f'{label_path}/', db_names=[db], labels=g_labels.node_labels,
                                         name=l.get_layer_string(), max_label_num=l.node_labels)
                    else:
                        # raise an error if the file does not exist and add the absolute path to the error message
                        raise FileNotFoundError(f"File {label_path} does not exist")
                    # add the properties to the graph data
                    if 'properties' in l.layer_dict:
                        prop_dict = l.layer_dict['properties']
                        prop_name = prop_dict['name']
                        if prop_name not in graph_data.properties:
                            graph_data.properties[prop_name] = Properties(path=properties_path, db_name=db,
                                                                          property_name=prop_dict['name'],
                                                                          valid_values=prop_dict['values'],
                                                                          layer_id=l.layer_id)
                        else:
                            graph_data.properties[prop_name].add_properties(prop_dict['values'], l.layer_id)
                    pass

                """
                    Get the first index in the results directory that is not used
                """
                para.set_file_index(size=6)

                net = RuleGNN.RuleGNN(graph_data=graph_data,
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