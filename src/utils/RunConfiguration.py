from platform import architecture

from click import option

from src.Architectures.RuleGNN.RuleGNNLayers import Layer
class RunConfiguration:
    def __init__(self, config, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, weight_decay, loss, task="classification"):
        self.config = config
        self.network_architecture = network_architecture.copy()
        self.layers = layers.copy()
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.loss = loss
        self.task = task

    def print(self):
        print(f"Network architecture: {self.network_architecture}")
        print(f"Layers: {self.layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Dropout: {self.dropout}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Loss: {self.loss}")

def generate_layer_options(layer_dict):
    options = []
    for label_type in layer_dict['labels']:
        properties_dict = []
        if layer_dict.get('properties', None) is not None:
            for prop_val in layer_dict['properties']:
                properties_dict.append(prop_val)
        key_list = list(label_type.keys())
        # remove label_type from key list
        key_list.remove('label_type')
        # remove all keys where the value is None or an empty list
        for key in key_list:
            if label_type[key] is None or (type(label_type[key]) == list and len(label_type[key]) == 0):
                key_list.remove(key)
        if len(key_list) == 0:
            if properties_dict is None or len(properties_dict) == 0:
                options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type}]})
            else:
                for prop_val in properties_dict:
                    options.append({'layer_type': layer_dict['layer_type'], 'channels': [{'labels': label_type, 'properties': prop_val.copy()}]})
        else:
            value_list = []
            for key in key_list:
                if type(label_type[key]) != list:
                    value_list.append([label_type[key]])
                else:
                    value_list.append(label_type[key])
            # get all value combinations as tuples over the value lists
            value_combinations = []
            for i in range(len(value_list)):
                if len(value_combinations) == 0:
                    for v in value_list[i]:
                        value_combinations.append([v])
                else:
                    new_combinations = []
                    for c in value_combinations:
                        for v in value_list[i]:
                            new_combinations.append(c + [v])
                    value_combinations = new_combinations
            for i, values in enumerate(value_combinations):
                if properties_dict is None or len(properties_dict) == 0:
                    curr_layer_dict = {'layer_type': layer_dict['layer_type']}
                    channels_list = []
                    label_dict = {'label_type': label_type['label_type']}
                    for j, value in enumerate(values):
                        label_dict[key_list[j]] = value
                    channels_list.append({'labels' : label_dict})
                    curr_layer_dict['channels'] = channels_list
                    options.append(curr_layer_dict)
                else:
                    for prop_val in properties_dict:
                        curr_layer_dict = {'layer_type': layer_dict['layer_type']}
                        channels_list = []
                        label_dict = {'label_type': label_type['label_type']}
                        for j, value in enumerate(values):
                            label_dict[key_list[j]] = value
                        channels_list.append({'labels' : label_dict, 'properties': prop_val.copy()})
                        curr_layer_dict['channels'] = channels_list
                        options.append(curr_layer_dict)
    return options

def get_network_architectures(network_architectures_dict: dict):
    network_architectures = []
    layers_per_architecture = []
    for network_architecture in network_architectures_dict:
        for i, layer in enumerate(network_architecture):
            if layer.get('labels', None) is not None and type(layer['labels']) == list:
                options = generate_layer_options(layer)
                if len(layers_per_architecture) <= i:
                    while len(layers_per_architecture) <= i:
                        layers_per_architecture.append([])
                for opt in options:
                    layers_per_architecture[i].append(opt)
            else:
                if len(layers_per_architecture) <= i:
                    while len(layers_per_architecture) <= i:
                        layers_per_architecture.append([])
                layers_per_architecture[i].append(network_architecture[i])
                break
    # get all possible network architectures using all combinations from layers per architecture
    for i in range(len(layers_per_architecture)):
        if len(network_architectures) == 0:
            for layer in layers_per_architecture[i]:
                network_architectures.append([layer])
        else:
            new_network_architectures = []
            for network_architecture in network_architectures:
                for layer in layers_per_architecture[i]:
                    new_network_architectures.append(network_architecture + [layer])
            network_architectures = new_network_architectures

    return network_architectures


def get_run_configs(experiment_configuration):
    # define the network type from the config file
    run_configs = []
    task = "classification"
    if 'task' in experiment_configuration:
        task = experiment_configuration['task']
    # get networks from the config file
    network_architectures = get_network_architectures(experiment_configuration['networks'])
    # iterate over all network architectures
    for network_architecture in network_architectures:
        layers = []
        # get all different run configurations
        for i, l in enumerate(network_architecture):
            layers.append(Layer(l, i))
        for b in experiment_configuration.get('batch_size', [128]):
            for lr in experiment_configuration.get('learning_rate', [0.001]):
                for e in experiment_configuration.get('epochs', [100]):
                    for d in experiment_configuration.get('dropout', [0.0]):
                        for o in experiment_configuration.get('optimizer', 'Adam'):
                            for w in experiment_configuration.get('weight_decay', [0.0]):
                                for loss in experiment_configuration.get('loss', ['CrossEntropyLoss']):
                                    run_configs.append(
                                        RunConfiguration(experiment_configuration, network_architecture, layers, b, lr, e, d, o, w, loss, task))
    return run_configs
