import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
import torch
from torch import optim, nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from src.utils import GraphData
from src.Architectures.RuleGNN import RuleGNN
from src.utils.Parameters import Parameters
from src.Time.TimeClass import TimeClass
from src.TrainTestData import TrainTestData as ttd
from src.utils.utils import get_k_lowest_nonzero_indices, valid_pruning_configuration, is_pruning

class EvaluationValues:
    def __init__(self):
        self.accuracy = 0.0
        self.accuracy_std = 0.0
        self.loss = 0.0
        self.loss_std = 0.0
        self.mae = 0.0
        self.mae_std = 0.0


class ModelEvaluation:
    def __init__(self, run_id: int, k_val: int, graph_data: GraphData.GraphData, model_data: Tuple[np.ndarray, np.ndarray, np.ndarray], seed: int, para: Parameters.Parameters):
        self.best_epoch = None
        self.device = None
        self.dtype = None
        self.run_id = run_id
        self.k_val = k_val
        self.graph_data = graph_data
        self.training_data, self.validate_data, self.test_data = model_data
        self.seed = seed
        self.para = para
        self.results_path = para.run_config.config['paths']['results']
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.net = None
        # get gpu or cpu: (cpu is recommended at the moment)
        if self.para.run_config.config.get('device', None) is not None:
            self.device = torch.device(self.para.run_config.config['device'] if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float
        if self.para.run_config.config.get('precision', 'float') == 'double':
            self.dtype = torch.double
            # set the inputs in graph_data to double precision
            self.graph_data.input_data = [x.double() for x in self.graph_data.input_data]

    def Run(self, run_seed: int = 687497):
        """
        Set up the network
        """

        self.net = RuleGNN.RuleGNN(graph_data=self.graph_data,
                              para=self.para,
                              seed=self.seed, device=self.device)
        # set the network to device
        self.net.to(self.device)

        timer = TimeClass()

        """
        Set up the loss function
        """
        if self.para.run_config.loss == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        elif self.para.run_config.loss in ['MeanSquaredError', 'MSELoss', 'mse', 'MSE']:
            self.criterion = nn.MSELoss()
        elif self.para.run_config.loss in ['L1Loss', 'l1', 'L1', 'mean_absolute_error', 'mae', 'MAE', 'MeanAbsoluteError']:
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        """
        Set up the optimizer
        """
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.para.learning_rate, weight_decay=self.para.run_config.weight_decay)

        self.preprocess_writer()

        """
        Variable learning rate
        """
        if self.para.run_config.config.get('scheduler', False):
            self.scheduler = StepLR(self.optimizer, step_size=10, gamma=0.5)

        """
        Store the best epoch
        """
        self.best_epoch = {"epoch": 0, "acc": 0.0, "loss": 1000000.0, "val_acc": 0.0, "val_loss": 1000000.0, "val_mae": 1000000.0}

        """
        Run through the defined number of epochs
        """
        seeds = np.arange(self.para.n_epochs*self.para.n_val_runs)
        seeds = np.reshape(seeds, (self.para.n_epochs, self.para.n_val_runs))
        for epoch in range(self.para.n_epochs):
            # Test stopping criterion
            if self.para.run_config.config['early_stopping']['enabled']:
                if epoch - self.best_epoch["epoch"] > self.para.run_config.config['early_stopping']['patience']:
                    if self.para.print_results:
                        print(f"Early stopping at epoch {epoch}")
                    break

            timer.measure("epoch")
            self.net.epoch = epoch
            epoch_values = EvaluationValues()
            validation_values = EvaluationValues()
            test_values = EvaluationValues()

            """
            Random Train batches for each epoch, run_id and k_val
            """
            shuffling_seed = seeds[epoch][self.k_val] * self.run_id + run_seed
            np.random.seed(shuffling_seed)
            np.random.shuffle(self.training_data)
            self.para.run_config.batch_size = min(self.para.run_config.batch_size, len(self.training_data))
            train_batches = np.array_split(self.training_data, self.training_data.size // self.para.run_config.batch_size)

            for batch_counter, batch in enumerate(train_batches, 0):
                timer.measure("forward")
                self.optimizer.zero_grad()
                outputs = Variable(torch.zeros((len(batch), self.graph_data.num_classes), dtype=self.dtype))
                outputs = outputs.to(self.device)
                labels = self.graph_data.output_data[batch].to(self.device)

                # TODO batch in one matrix ?
                self.net.train(True)
                for j, graph_id in enumerate(batch, 0):
                    timer.measure("forward_step")
                    if self.para.run_config.config.get('input_features', None).get('random_variation', None):
                        mean = self.para.run_config.config['input_features']['random_variation'].get('mean', 0.0)
                        std = self.para.run_config.config['input_features']['random_variation'].get('std', 0.1)
                        # random variation as torch tensor
                        random_variation = np.random.normal(mean, std, self.graph_data.input_data[graph_id].shape)
                        if self.para.run_config.config.get('precision', 'double') == 'float':
                            random_variation = torch.FloatTensor(random_variation)
                        else:
                            random_variation = torch.DoubleTensor(random_variation)
                        network_input = self.graph_data.input_data[graph_id] + random_variation
                    else:
                        network_input = self.graph_data.input_data[graph_id]
                    outputs[j] = self.net(network_input.to(self.device), graph_id)
                    timer.measure("forward_step")


                loss = self.criterion(outputs, labels)
                timer.measure("forward")

                weights = []
                if self.para.save_weights:
                    for i, layer in enumerate(self.net.net_layers):
                        weights.append([x.item() for x in layer.Param_W])
                        w = np.array(weights[-1]).reshape(1, -1)
                        #df = pd.DataFrame(w)
                        #df.to_csv(f"Results/Parameter/layer_{i}_weights.csv", header=False, index=False, mode='a')

                timer.measure("backward")
                # change learning rate with high loss
                #for g in optimizer.param_groups:
                #    loss_value = loss.item()
                #    min_val = 50 - epoch ** (1. / 6.) * (49 / self.para.n_epochs ** (1. / 6.))
                #    loss_val = 100 * loss_value ** 2
                #learning_rate_mul = min(min_val, loss_val)
                #g['lr'] = self.para.learning_rate * learning_rate_mul
                # print min_val, loss_val, learning_rate_mul, g['lr']
                #    if self.para.print_results:
                #        print(f'Min: {min_val}, Loss: {loss_val}, Learning rate: {g["lr"]}')

                loss.backward()
                self.optimizer.step()
                timer.measure("backward")
                timer.reset()

                if self.para.save_weights:
                    weight_changes = []
                    for i, layer in enumerate(self.net.net_layers):
                        change = np.array(
                            [weights[i][j] - x.item() for j, x in enumerate(layer.Param_W)]).flatten().reshape(1, -1)
                        weight_changes.append(change)
                        # save to three differen csv files using pandas
                        #df = pd.DataFrame(change)
                        #df.to_csv(f'Results/Parameter/layer_{i}_change.csv', header=False, index=False, mode='a')
                        # if there is some change print that the layer trains
                        if np.count_nonzero(change) > 0:
                            print(f'Layer {i} has updated')
                        else:
                            print(f'Layer {i} has not updated')

                epoch_values.loss += loss.item()

                '''
                Evaluate the training accuracy
                '''
                epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='training', outputs=outputs, labels=labels,  batch_idx=batch_counter, batch_length=len(batch), num_batches=len(train_batches))


            '''
            Pruning
            '''

            if valid_pruning_configuration(self.para, epoch):
                self.model_pruning(epoch)


            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='validation')
            epoch_values, validation_values, test_values = self.evaluate_results(epoch=epoch,train_values=epoch_values, validation_values=validation_values, test_values=test_values, evaluation_type='test')

            timer.measure("epoch")
            epoch_time = timer.get_flag_time("epoch")

            self.postprocess_writer(epoch, epoch_time, epoch_values, validation_values, test_values)


            if self.para.run_config.config.get('scheduler', False):
                # check if learning rate is > 0.0001
                if self.optimizer.param_groups[0]['lr'] > 0.0001:
                    self.scheduler.step()

    def model_pruning(self, epoch):
        # prune each five epochs

        print('Pruning')
        # iterate over the layers of the neural net
        for i, layer in enumerate(self.net.net_layers):
            pruning_per_layer = self.para.run_config.config['prune']['percentage'][i]
            # use total number of epochs, the epoch step and the pruning percentage
            pruning_per_layer /= (self.para.n_epochs / self.para.run_config.config['prune']['epochs']) - 1

            # get tensor from the parameter_list layer.Param_W
            layer_tensor = torch.abs(torch.tensor(layer.Param_W) * torch.tensor(layer.mask))
            # print number of non zero entries in layer_tensor
            print(f'Number of non zero entries in before pruning {layer.name}: {torch.count_nonzero(layer_tensor)}')
            # get the indices of the trainable parameters with lowest absolute max(1, 1%)
            k = int(layer_tensor.size(0) * pruning_per_layer)
            if k != 0:
                low = torch.topk(layer_tensor, k, largest=False)
                lowest_indices = get_k_lowest_nonzero_indices(layer_tensor, k)
                # set all indices in layer.mask to zero
                layer.mask[lowest_indices] = 0
                layer.Param_W.data = layer.Param_W_original * layer.mask
                # for c, graph_weight_distribution in enumerate(layer.weight_distribution):
                #     new_graph_weight_distribution = None
                #     for [i, j, pos] in graph_weight_distribution:
                #         # if pos is in lowest_indices do nothing else append to new_graph_weight_distribution
                #         if pos in lowest_indices:
                #             pass
                #         else:
                #             if new_graph_weight_distribution is None:
                #                 new_graph_weight_distribution = np.array([i, j, pos])
                #             else:
                #                 # add [i, j, pos] to new_graph_weight_distribution
                #                 new_graph_weight_distribution = np.vstack((new_graph_weight_distribution, [i, j, pos]))
                #     layer.weight_distribution[c] = new_graph_weight_distribution

            # print number of non zero entries in layer.Param_W
            print(
                f'Number of non zero entries in layer after pruning {layer.name}: {torch.count_nonzero(layer.Param_W)}')
        if is_pruning(self.para.run_config.config):
            for i, layer in enumerate(self.net.net_layers):
                # get tensor from the parameter_list layer.Param_W
                layer_tensor = torch.abs(torch.tensor(layer.Param_W).clone().detach() * torch.tensor(layer.mask))
                # print number of non zero entries in layer_tensor
                print(
                    f'Number of non zero entries in layer {layer.name}: {torch.count_nonzero(layer_tensor)}/{torch.numel(layer_tensor)}')

                # multiply the Param_W with the mask
                layer.Param_W.data = layer.Param_W.data * layer.mask

    def preprocess_writer(self):
        if self.run_id == 0 and self.k_val == 0:
            # create a file about the net details including (net, optimizer, learning rate, loss function, batch size, number of classes, number of epochs, balanced data, dropout)
            file_name = f'{self.para.db}_{self.para.config_id}_Network.txt'
            final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
            with open(final_path, "a") as file_obj:
                file_obj.write(f"Network architecture: {self.para.run_config.network_architecture}\n"
                               f"Optimizer: {self.optimizer}\n"
                               f"Loss function: {self.criterion}\n"
                               f"Batch size: {self.para.batch_size}\n"
                               f"Balanced data: {self.para.balance_data}\n"
                               f"Number of epochs: {self.para.n_epochs}\n")
                # iterate over the layers of the neural net
                total_trainable_parameters = 0
                for layer in self.net.net_layers:
                    file_obj.write(f"\n")
                    try:
                        file_obj.write(f"Layer: {layer.name}\n")
                    except:
                        file_obj.write(f"Linear Layer\n")
                    file_obj.write(f"\n")
                    # get number of trainable parameters
                    layer_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                    file_obj.write(f"Trainable Parameters: {layer_params}\n")
                    try:
                        file_obj.write(f"Node labels: {layer.node_labels.num_unique_node_labels}\n")
                    except:
                        pass
                    try:
                        for i, n in enumerate(layer.n_properties):
                            file_obj.write(f"Number of pairwise properties in channel {i}: {n}\n")
                    except:
                        pass
                    weight_learnable_parameters = 0
                    bias_learnable_parameters = 0
                    try:
                        if layer.Param_W.requires_grad:
                            total_trainable_parameters += layer.Param_W.numel()
                            weight_learnable_parameters += layer.Param_W.numel()
                    except:
                        pass
                    try:
                        if layer.Param_b.requires_grad:
                            total_trainable_parameters += layer.Param_b.numel()
                            bias_learnable_parameters += layer.Param_b.numel()
                    except:
                        pass

                    file_obj.write("Weight matrix learnable parameters: {}\n".format(weight_learnable_parameters))
                    file_obj.write("Bias learnable parameters: {}\n".format(bias_learnable_parameters))
                    try:
                        file_obj.write(f"Edge labels: {layer.edge_labels.num_unique_edge_labels}\n")
                    except:
                        pass
                for name, param in self.net.named_parameters():
                    file_obj.write(f"Layer: {name} -> {param.requires_grad}\n")

                file_obj.write(f"\n")
                file_obj.write(f"Total trainable parameters: {total_trainable_parameters}\n")

        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'

        does_run_exist = False
        # check if the file already exists
        if Path(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')).exists():
            # load the file with pandas and get the last epoch completed
            df = pd.read_csv(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}'), delimiter=';')
            if df['Epoch'].size <= 1:
                last_epoch = 0
            else:
                last_epoch = df['Epoch'].iloc[-1]
            # if the last_epoch equals the number of epochs the run is already completed
            if last_epoch != self.para.run_config.epochs - 1:
                does_run_exist = False
                print(f'The file {file_name} already exists but the run was not completed, or new parameters are used')
            else:
                does_run_exist = True
                print(f'The file {file_name} already exists and recomputation is skipped')

        if does_run_exist:
            return
        else:
            # if the file does not exist create a new file
            with open(self.results_path.joinpath(f'{self.para.db}/Results/{file_name}'), "w") as file_obj:
                file_obj.write("")

        # header use semicolon as delimiter
        if self.para.run_config.task == 'regression':
            header = "Dataset;RunNumber;ValidationNumber;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;" \
                     "EpochAccuracy;EpochTime;EpochMAE;EpochMAEStd;ValidationLoss;ValidationAccuracy;ValidationMAE;ValidationMAEStd;TestLoss;TestAccuracy;TestMAE;TestMAEStd\n"
        else:
            header = "Dataset;RunNumber;ValidationNumber;Epoch;TrainingSize;ValidationSize;TestSize;EpochLoss;EpochAccuracy;" \
                     "EpochTime;ValidationAccuracy;ValidationLoss;TestAccuracy;TestLoss\n"

        # Save file for results and add header if the file is new
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            if os.stat(final_path).st_size == 0:
                file_obj.write(header)

    def evaluate_results(self, epoch: int, train_values: EvaluationValues, validation_values: EvaluationValues, test_values: EvaluationValues, evaluation_type, outputs=None, labels=None, batch_idx=0, batch_length=0, num_batches=0):
        if evaluation_type == 'training':
            batch_acc = 0
            if self.para.run_config.config == 'classification':
                batch_acc = 100 * ttd.get_accuracy(outputs, labels, one_hot_encoding=True)
                train_values.accuracy += batch_acc * (batch_length / len(self.training_data))
            # if num classes is one calculate the mae and mae_std or if the task is regression
            elif self.para.run_config.task == 'regression':
                # flatten the labels and outputs
                flatten_labels = labels.detach().clone().flatten()
                flatten_outputs = outputs.detach().clone().flatten()
                if self.para.run_config.config.get('output_features_inverse', None) is not None:
                    flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config['output_features_inverse'])
                    flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config['output_features_inverse'])
                batch_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                batch_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                train_values.mae += batch_mae * (batch_length / len(self.training_data))
                train_values.mae_std += batch_mae_std * (batch_length / len(self.training_data))

            if self.para.print_results:
                if self.graph_data.num_classes == 1 or self.para.run_config.task == 'regression':
                    print(
                        "\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} %, mae: {}, mae_std: {}".format(epoch + 1,
                                                                                                         self.para.n_epochs,
                                                                                                         batch_idx + 1,
                                                                                                         num_batches,
                                                                                                         train_values.running_loss,
                                                                                                         batch_acc,
                                                                                                         train_values.mae,
                                                                                                         train_values.mae_std))
                else:
                    print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} % ".format(epoch + 1, self.para.n_epochs,
                                                                                      batch_idx + 1,
                                                                                      num_batches,
                                                                                      train_values.running_loss, batch_acc))
            self.para.count += 1

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/training_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'validation':
            '''
            Evaluate the validation accuracy for each epoch
            '''
            if self.validate_data.size != 0:
                outputs = torch.zeros((len(self.validate_data), self.graph_data.num_classes), dtype=self.dtype)
                labels = self.graph_data.output_data[self.validate_data]

                # use torch no grad to save memory
                with torch.no_grad():
                    for j, data_pos in enumerate(self.validate_data):
                        self.net.train(False)
                        outputs[j] = self.net(self.graph_data.input_data[data_pos].to(self.device), data_pos)

                # get validation loss
                validation_loss = self.criterion(outputs, labels).item()
                validation_values.loss = validation_loss
                if self.para.run_config.task == 'regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    validation_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae = validation_mae
                    validation_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    validation_values.mae_std = validation_mae_std
                else:
                    labels_argmax = labels.argmax(axis=1)
                    outputs_argmax = outputs.argmax(axis=1)
                    validation_acc = 100 * sklearn.metrics.accuracy_score(labels_argmax, outputs_argmax)
                    validation_values.accuracy = validation_acc

                # update best epoch
                if self.para.run_config.task == 'regression':
                    if validation_values.mae <= self.best_epoch["val_mae"] or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_loss"] = validation_values.loss
                        self.best_epoch["val_mae"] = validation_values.mae
                        self.best_epoch["val_mae_std"] = validation_values.mae_std
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if 'best_model' in self.para.run_config.config and self.para.run_config.config['best_model']:
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(),final_path)


                else:
                    # check if pruning is on, then save the best model in the last pruning epoch
                    if (validation_values.accuracy > self.best_epoch["val_acc"] or validation_values.accuracy == self.best_epoch[
                        "val_acc"] and validation_loss < self.best_epoch["val_loss"]) or valid_pruning_configuration(self.para, epoch):
                        self.best_epoch["epoch"] = epoch
                        self.best_epoch["acc"] = train_values.accuracy
                        self.best_epoch["loss"] = train_values.loss
                        self.best_epoch["val_acc"] = validation_values.accuracy
                        self.best_epoch["val_loss"] = validation_values.loss
                        # save the best model
                        best_model_path = self.results_path.joinpath(f'{self.para.db}/Models/')
                        if not os.path.exists(best_model_path):
                            os.makedirs(best_model_path)
                        # Save the model if best model is used
                        if 'best_model' in self.para.run_config.config and self.para.run_config.config['best_model']:
                            final_path = self.results_path.joinpath(f'{self.para.db}/Models/model_{self.para.config_id}_run_{self.run_id}_val_step_{self.k_val}.pt')
                            torch.save(self.net.state_dict(), final_path)

            if self.para.save_prediction_values:
                # print outputs and labels to a csv file
                outputs_np = outputs.detach().numpy()
                # transpose the numpy array
                outputs_np = outputs_np.T
                df = pd.DataFrame(outputs_np)
                # show only two decimal places
                df = df.round(2)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')
                labels_np = labels.detach().numpy()
                labels_np = labels_np.T
                df = pd.DataFrame(labels_np)
                df.to_csv("Results/Parameter/validation_predictions.csv", header=False, index=False, mode='a')

        elif evaluation_type == 'test':
            # Test accuracy
            # print only if run best model is used
            if self.para.run_config.config.get('best_model', False):
                # Test accuracy
                outputs = torch.zeros((len(self.test_data), self.graph_data.num_classes), dtype=self.dtype)
                labels = self.graph_data.output_data[self.test_data]

                with torch.no_grad():
                    for j, data_pos in enumerate(self.test_data, 0):
                        self.net.train(False)
                        outputs[j] = self.net(self.graph_data.input_data[data_pos].to(self.device), data_pos)


                test_loss = self.criterion(outputs, labels).item()
                test_values.loss = test_loss
                if self.para.run_config.task == 'regression':
                    flatten_labels = labels.detach().clone().flatten()
                    flatten_outputs = outputs.detach().clone().flatten()
                    if self.para.run_config.config.get('output_features_inverse', None) is not None:
                        flatten_labels = GraphData.transform_data(flatten_labels, self.para.run_config.config[
                            'output_features_inverse'])
                        flatten_outputs = GraphData.transform_data(flatten_outputs, self.para.run_config.config[
                            'output_features_inverse'])
                    test_mae = torch.mean(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae = test_mae
                    test_mae_std = torch.std(torch.abs(flatten_labels - flatten_outputs))
                    test_values.mae_std = test_mae_std
                else:
                    test_acc = 100 * sklearn.metrics.accuracy_score(labels.argmax(axis=1), outputs.argmax(axis=1))
                    test_values.accuracy = test_acc

                if self.para.print_results:
                    np_labels = labels.detach().numpy()
                    np_outputs = outputs.detach().numpy()
                    # np array of correct/incorrect predictions
                    labels_argmax = np_labels.argmax(axis=1)
                    outputs_argmax = np_outputs.argmax(axis=1)
                    # change if task is regression
                    if 'task' in self.para.run_config.config and self.para.run_config.config['task'] == 'regression':
                        np_correct = np_labels - np_outputs
                    else:
                        np_correct = labels_argmax == outputs_argmax
                    # print entries of np_labels and np_outputs
                    for j, data_pos in enumerate(self.test_data, 0):
                        print(data_pos, np_labels[j], np_outputs[j], np_correct[j])

                if self.para.save_prediction_values:
                    # print outputs and labels to a csv file
                    outputs_np = outputs.detach().numpy()
                    # transpose the numpy array
                    outputs_np = outputs_np.T
                    df = pd.DataFrame(outputs_np)
                    # show only two decimal places
                    df = df.round(2)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')
                    labels_np = labels.detach().numpy()
                    labels_np = labels_np.T
                    df = pd.DataFrame(labels_np)
                    df.to_csv("Results/Parameter/test_predictions.csv", header=False, index=False, mode='a')

        return train_values, validation_values, test_values

    def postprocess_writer(self, epoch, epoch_time, train_values: EvaluationValues, validation_values: EvaluationValues, test_values: EvaluationValues):
        if self.para.print_results:
            # if class num is one print the mae and mse
            if self.para.run_config.task == 'regression':
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy} epoch mae: {train_values.mae} +- {train_values.mae_std} epoch time: {epoch_time}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss} validation mae: {validation_values.mae} +- {validation_values.mae_std}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss} test mae: {test_values.mae} +- {test_values.mae_std}'
                    f'time: {epoch_time}')
            else:
                print(
                    f'run: {self.run_id} val step: {self.k_val} epoch: {epoch + 1}/{self.para.n_epochs} epoch loss: {train_values.loss} epoch acc: {train_values.accuracy}'
                    f' validation acc: {validation_values.accuracy} validation loss: {validation_values.loss}'
                    f'test acc: {test_values.accuracy} test loss: {test_values.loss}'
                    f'time: {epoch_time}')

        if self.para.run_config.task == 'regression':
            res_str = f"{self.para.db};{self.run_id};{self.k_val};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                      f"{train_values.loss};{train_values.accuracy};{epoch_time};{train_values.mae};{train_values.mae_std};" \
                        f"{validation_values.loss};{validation_values.accuracy};{validation_values.mae};{validation_values.mae_std};" \
                        f"{test_values.loss};{test_values.accuracy};{test_values.mae};{test_values.mae_std}\n"
        else:
            res_str = f"{self.para.db};{self.run_id};{self.k_val};{epoch};{self.training_data.size};{self.validate_data.size};{self.test_data.size};" \
                      f"{train_values.loss};{train_values.accuracy};{epoch_time};{validation_values.accuracy};{validation_values.loss};{test_values.accuracy};{test_values.loss}\n"

        # Save file for results
        file_name = f'{self.para.db}_{self.para.config_id}_Results_run_id_{self.run_id}_validation_step_{self.para.validation_id}.csv'
        final_path = self.results_path.joinpath(f'{self.para.db}/Results/{file_name}')
        with open(final_path, "a") as file_obj:
            file_obj.write(res_str)

        if self.para.draw:
            self.para.draw_data = ttd.plot_learning_data(epoch + 1,
                                                         [train_values.accuracy, validation_values.accuracy, test_values.accuracy, train_values.loss],
                                                         self.para.draw_data, self.para.n_epochs)


