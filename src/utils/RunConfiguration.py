from src.Architectures.RuleGNN.RuleGNNLayers import Layer
class RunConfiguration():
    def __init__(self, network_architecture, layers, batch_size, lr, epochs, dropout, optimizer, loss,
                 task="classification"):
        self.network_architecture = network_architecture
        self.layers = layers
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.task = "classification"

    def print(self):
        print(f"Network architecture: {self.network_architecture}")
        print(f"Layers: {self.layers}")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.lr}")
        print(f"Epochs: {self.epochs}")
        print(f"Dropout: {self.dropout}")
        print(f"Optimizer: {self.optimizer}")
        print(f"Loss: {self.loss}")


def get_run_configs(configs):
    # define the network type from the config file
    run_configs = []
    task = "classification"
    if 'task' in configs:
        task = configs['task']
    # iterate over all network architectures
    for network_architecture in configs['networks']:
        layers = []
        # get all different run configurations
        for i, l in enumerate(network_architecture):
            layers.append(Layer(l, i))
        for b in configs['batch_size']:
            for lr in configs['learning_rate']:
                for e in configs['epochs']:
                    for d in configs['dropout']:
                        for o in configs['optimizer']:
                            for loss in configs['loss']:
                                run_configs.append(
                                    RunConfiguration(network_architecture, layers, b, lr, e, d, o, loss, task))
    return run_configs
