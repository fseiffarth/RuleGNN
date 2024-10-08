from src.Architectures.RuleGNN.RuleGNNLayers import Layer
class RunConfiguration:
    def __init__(self, config, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, weight_decay, loss, task="classification"):
        self.config = config
        self.network_architecture = network_architecture
        self.layers = layers
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


def get_run_configs(experiment_configuration):
    # define the network type from the config file
    run_configs = []
    task = "classification"
    if 'task' in experiment_configuration:
        task = experiment_configuration['task']
    # iterate over all network architectures
    for network_architecture in experiment_configuration['networks']:
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
