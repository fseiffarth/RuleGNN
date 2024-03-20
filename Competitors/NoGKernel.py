import os
from typing import List

import numpy as np

from GraphData import NodeLabeling, EdgeLabeling
from GraphData.GraphData import GraphData
from Parameters.Parameters import Parameters
from sklearn.svm import SVC


class NoGKernel():
    def __init__(self, graph_data:GraphData, training_data:List[int], validate_data:List[int], test_data:List[int], seed:int, results_path:str):
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.results_path = results_path

    def Run(self):
        # create numpy vector from the graph data labels
        unique_node_labels = self.graph_data.node_labels['primary']
        unique_edge_labels = self.graph_data.edge_labels['primary']
        X = np.zeros(shape=(self.graph_data.num_graphs, unique_node_labels.num_unique_node_labels + unique_edge_labels.num_unique_edge_labels))
        # fill the numpy vector with number of unique node and edge labels
        for i in range(0, self.graph_data.num_graphs):
            for j in range(0, unique_node_labels):
                if j in self.graph_data.unique_node_labels[i]:
                    X[i, j] = self.graph_data.unique_node_labels[i][j]
            for j in range(self.graph_data.num_unique_node_labels, self.graph_data.num_unique_node_labels + self.graph_data.num_unique_edge_labels):
                if j - self.graph_data.num_unique_node_labels in self.graph_data.unique_edge_labels[i]:
                    X[i, j] = self.graph_data.unique_edge_labels[i][j - self.graph_data.num_unique_node_labels]

        Y = np.asarray(self.graph_data.graph_labels)
        # split the data in training, validation and test set
        X_train = X[self.training_data]
        Y_train = Y[self.training_data]
        X_val = X[self.validate_data]
        Y_val = Y[self.validate_data]
        X_test = X[self.test_data]
        Y_test = Y[self.test_data]

        # create a SVM based on an RBF kernel that trains on the training data
        # and predicts the labels of the validation data and test data
        clf = SVC(kernel='rbf', C=1)
        clf.fit(X_train, Y_train)

        Y_test_pred = clf.predict(X_test)
        # calculate the accuracy of the prediction
        test_acc = np.mean(Y_test_pred == Y_test)
        val_acc = 0
        # check if X_val is not empty
        if len(X_val) != 0:
            Y_val_pred = clf.predict(X_val)
            val_acc = np.mean(Y_val_pred == Y_val)
            # print the accuracy of the prediction
            print(f"Run {self.run} Validation Step: {self.k_val} Validation Accuracy: {val_acc} Test Accuracy: {test_acc}")
        else:
            print(f"Run {self.run} Validation Step: {self.k_val} Test Accuracy: {test_acc}")
        # save results to Results/NoGKernel/NoGKernelResults.csv if NoGKernel folder does not exist create it
        if not os.path.exists(f"{self.results_path}/NoGKernel"):
            os.makedirs(f"{self.results_path}/NoGKernel")
        with open(f"{self.results_path}/NoGKernel/Results.csv", 'a') as f:
            # if file is empty write the header
            if os.stat(f"{self.results_path}/NoGKernel/Results.csv").st_size == 0:
                f.write("Run,Validation Step,Validation Accuracy,Test Accuracy\n")
            f.write(f"{self.run},{self.k_val},{val_acc},{test_acc}\n")




