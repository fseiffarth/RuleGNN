import os
from typing import List

import numpy as np
import torch

from GraphData.GraphData import GraphData
from utils.Parameters import Parameters


class NoGKernelNN():
    def __init__(self, run:int, k_val:int, graph_data:GraphData, training_data:List[int], validate_data:List[int], test_data:List[int], seed:int, para:Parameters, results_path:str):
        self.run = run
        self.k_val = k_val
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.para = para
        self.results_path = results_path

    def Run(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # create numpy vector from the graph data labels
        num_unique_node_labels = self.graph_data.primary_node_labels.num_unique_node_labels
        num_unique_edge_labels = self.graph_data.primary_edge_labels.num_unique_edge_labels
        X = np.zeros(shape=(self.graph_data.num_graphs, num_unique_node_labels + num_unique_edge_labels))
        # fill the numpy vector with number of unique node and edge labels
        for i in range(0, self.graph_data.num_graphs):
            for j in range(0, num_unique_node_labels):
                if j in self.graph_data.primary_node_labels.unique_node_labels[i]:
                    X[i, j] = self.graph_data.primary_node_labels.unique_node_labels[i][j]
            for j in range(num_unique_node_labels, num_unique_node_labels + num_unique_edge_labels):
                if j - num_unique_node_labels in self.graph_data.primary_edge_labels.unique_edge_labels[i]:
                    X[i, j] = self.graph_data.primary_edge_labels.unique_edge_labels[i][j - num_unique_node_labels]

        Y = np.asarray(self.graph_data.graph_labels)
        # split the data in training, validation and test set
        X_train = X[self.training_data]
        Y_train = Y[self.training_data]
        X_val = X[self.validate_data]
        Y_val = Y[self.validate_data]
        X_test = X[self.test_data]
        Y_test = Y[self.test_data]

        # convert the arrays to tensors
        X_train = torch.tensor(X_train).float().to(device)
        Y_train = torch.tensor(Y_train)
        # Y_train class -1 to 0
        Y_train[Y_train == -1] = 0
        Y_train_one_hot = torch.nn.functional.one_hot(Y_train).float().to(device)
        X_val = torch.tensor(X_val)
        Y_val = torch.tensor(Y_val)
        X_test = torch.tensor(X_test).to(device)
        Y_test = torch.tensor(Y_test)
        Y_test[Y_test == -1] = 0
        Y_test_one_hot = torch.nn.functional.one_hot(Y_test).float().to(device)

        net = self.NeuralNet(n_input=X_train.shape[1], n_hidden=X_train.shape[1], n_output=2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = net(X_train)
            loss = criterion(outputs, Y_train_one_hot)
            print(f"Epoch {epoch} Loss: {loss.item()}")
            loss.backward()
            optimizer.step()

        # evaluate the performance of the model on the training data
        outputs = net(X_train.float())
        train_acc = np.mean(outputs.argmax(1).cpu().numpy() == Y_train_one_hot.argmax(1).cpu().numpy())
        # evaluate the performance of the model on the test data
        outputs = net(X_test.float())
        test_acc = np.mean(outputs.argmax(1).cpu().numpy() == Y_test_one_hot.argmax(1).cpu().numpy())
        val_acc = 0

        # save results to Results/NoGKernel/NoGKernelResults.csv if NoGKernel folder does not exist create it
        if not os.path.exists(f"{self.results_path}/NoGKernel"):
            os.makedirs(f"{self.results_path}/NoGKernel")
        with open(f"{self.results_path}/NoGKernel/ResultsNN.csv", 'a') as f:
            # if file is empty write the header
            if os.stat(f"{self.results_path}/NoGKernel/ResultsNN.csv").st_size == 0:
                f.write("Run,Validation Step, Train Accuracy, Validation Accuracy,Test Accuracy\n")
            f.write(f"{self.run},{self.k_val}, {train_acc}, {val_acc},{test_acc}\n")


    # def simple neural network with two hidden layers using torch
    class NeuralNet(torch.nn.Module):
        def __init__(self, n_input, n_hidden, n_output):
            super(NoGKernelNN.NeuralNet, self).__init__()
            self.hidden1 = torch.nn.Linear(n_input, n_hidden)
            self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
            self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
            self.out = torch.nn.Linear(n_hidden, n_output)

        def forward(self, x):
            x = torch.tanh(self.hidden1(x))
            x = torch.tanh(self.hidden2(x))
            x = torch.tanh(self.hidden3(x))
            x = torch.tanh(self.out(x))
            return x


