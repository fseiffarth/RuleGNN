import os

from src.Architectures.RuleGNN.RuleGNNLayers import Layer, get_label_string
from src.Preprocessing.create_labels import save_node_labels
from src.utils.GraphData import GraphData
from src.utils.GraphLabels import combine_node_labels, Properties
from src.utils.Parameters.Parameters import Parameters
from src.utils.load_labels import load_labels
from src.utils import ReadWriteGraphs as gdtgl


def load_preprocessed_data_and_parameters(run_id, validation_id, config_id, validation_folds, graph_data:GraphData, run_config, para: Parameters):
    experiment_configuration = run_config.config
    # path do db and db
    draw = False
    print_results = False
    save_weights = False
    save_prediction_values = False
    plot_graphs = False
    print_layer_init = False
    # if debug mode is on, turn on all print and draw options
    if experiment_configuration['mode'] == "debug":
        draw = experiment_configuration['additional_options']['draw']
        print_results = experiment_configuration['additional_options']['print_results']
        save_prediction_values = experiment_configuration['additional_options']['save_prediction_values']
        save_weights = experiment_configuration['additional_options']['save_weights']
        plot_graphs = experiment_configuration['additional_options']['plot_graphs']

    unique_label_dicts = []
    label_layer_ids = []
    unique_properties = []
    properties_layer_ids = []
    for i, l in enumerate(run_config.layers):
        # add the labels to the graph data
        new_unique = l.get_unique_layer_dicts()
        for x in new_unique:
            if x not in unique_label_dicts:
                unique_label_dicts.append(x)
        property_dicts = l.get_unique_property_dicts()
        if property_dicts:
            for x in property_dicts:
                if x["name"] not in unique_properties:
                    unique_properties.append(x["name"])

    for label_dict in unique_label_dicts:
        label_path = experiment_configuration['paths']['labels'].joinpath(f"{graph_data.graph_db_name}_{get_label_string(label_dict)}_labels.txt")
        if os.path.exists(label_path):
            g_labels = load_labels(path=label_path)
            graph_data.node_labels[get_label_string(label_dict)] = g_labels
        elif l.layer_type == "combined":  # create combined file if it is a combined layer and the file does not exist
            combined_labels = []
            # get the labels for each layer in the combined layer
            for x in l.layer_dict['sub_labels']:
                sub_layer = Layer(x, i)
                sub_label_path = experiment_configuration['paths']['labels'].joinpath(f"/{graph_data.graph_db_name}_{sub_layer.get_layer_string()}_labels.txt")
                if os.path.exists(sub_label_path):
                    g_labels = load_labels(path=sub_label_path)
                    combined_labels.append(g_labels)
                else:
                    # raise an error if the file does not exist
                    raise FileNotFoundError(f"File {sub_label_path} does not exist")
            # combine the labels and save them
            g_labels = combine_node_labels(combined_labels)
            graph_data.node_labels[l.get_layer_string()] = g_labels
            save_node_labels(graph_data=graph_data, labels=g_labels.node_labels,
                             label_path=label_path,
                             label_string=l.get_layer_string(), max_label_num=l.node_labels)
        else:
            # raise an error if the file does not exist and add the absolute path to the error message
            raise FileNotFoundError(f"File {label_path} does not exist")

    for prop_name in unique_properties:
        valid_values = {}
        for i, l in enumerate(run_config.layers):
            for j, c in enumerate(l.layer_channels):
                if c.property_dict is not None:
                    if c.property_dict.get('name', None) == prop_name:
                        valid_values[(i,j)] = c.property_dict.get('values', None)
        graph_data.properties[prop_name] = Properties(path=experiment_configuration['paths']['properties'], db_name=graph_data.graph_db_name,
                                                      property_name=prop_name,
                                                      valid_values=valid_values)

    """
        BenchmarkGraphs parameters
    """
    para.set_data_param(db=graph_data.graph_db_name,
                        max_coding=1,
                        layers=run_config.layers, node_features=1,
                        run_config=run_config)

    """
        Network parameters
    """
    para.set_evaluation_param(run_id=run_id, n_val_runs=validation_folds,
                              validation_id=validation_id,
                              config_id=config_id,
                              n_epochs=run_config.epochs,
                              learning_rate=run_config.lr,
                              dropout=run_config.dropout,
                              balance_data=run_config.config['balance_training'],
                              convolution_grad=True,
                              resize_graph=True)

    """
    Print, save and draw parameters
    """
    para.set_print_param(no_print=False, print_results=print_results, net_print_weights=True, print_number=1,
                         draw=draw, save_weights=save_weights,
                         save_prediction_values=save_prediction_values, plot_graphs=plot_graphs,
                         print_layer_init=print_layer_init)

    """
        Get the first index in the results directory that is not used
    """
    para.set_file_index(size=6)

    if para.plot_graphs:
        # if not exists create the directory
        if not os.path.exists(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots")):
            os.makedirs(experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots"))
        for i in range(0, len(graph_data.graphs)):
            gdtgl.draw_graph(graph_data.graphs[i], graph_data.graph_labels[i],
                             experiment_configuration['paths']['results'].joinpath(f"{para.db}/Plots/graph_{str(i).zfill(5)}.png"))
