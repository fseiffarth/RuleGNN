import os
from typing import List

import numpy as np

from GraphData import NodeLabeling, EdgeLabeling
from GraphData.GraphData import GraphData
from Parameters.Parameters import Parameters
from sklearn.svm import SVC


class NoGKernel():
    def __init__(self, graph_data: GraphData, run_num:int, validation_num:int, training_data: List[int], validate_data: List[int], test_data: List[int],
                 seed: int):
        self.graph_data = graph_data
        self.training_data = training_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.seed = seed
        self.run_num = run_num
        self.validation_num = validation_num

    def Run(self):
        # create numpy vector from the graph data labels
        primary_node_labels = self.graph_data.node_labels['primary']
        primary_edge_labels = self.graph_data.edge_labels['primary']
        X = np.zeros(shape=(self.graph_data.num_graphs,
                            primary_node_labels.num_unique_node_labels + primary_edge_labels.num_unique_edge_labels))
        # fill the numpy vector with number of unique node and edge labels
        for i in range(0, self.graph_data.num_graphs):
            for j in range(0, primary_node_labels.num_unique_node_labels):
                if j in primary_node_labels.unique_node_labels[i]:
                    X[i, j] = primary_node_labels.unique_node_labels[i][j]
            for j in range(primary_node_labels.num_unique_node_labels,
                           primary_node_labels.num_unique_node_labels + primary_edge_labels.num_unique_edge_labels):
                if j - primary_node_labels.num_unique_node_labels in primary_edge_labels.unique_edge_labels[i]:
                    X[i, j] = primary_edge_labels.unique_edge_labels[i][j - primary_node_labels.num_unique_node_labels]

        Y = np.asarray(self.graph_data.graph_labels)
        # split the data in training, validation and test set
        X_train = X[self.training_data]
        Y_train = Y[self.training_data]
        X_val = X[self.validate_data]
        Y_val = Y[self.validate_data]
        X_test = X[self.test_data]
        Y_test = Y[self.test_data]

        for c_param in [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11, 13, 15]:
            c_param = 2 ** c_param
            # create a SVM based on an RBF kernel that trains on the training data
            # and predicts the labels of the validation data and test data
            clf = SVC(kernel='rbf', C=c_param, random_state=self.seed)
            clf.fit(X_train, Y_train)
            Y_val_pred = clf.predict(X_val)
            Y_test_pred = clf.predict(X_test)
            # calculate the accuracy of the prediction
            val_acc = np.mean(Y_val_pred == Y_val)
            test_acc = np.mean(Y_test_pred == Y_test)

            file_name = f'{self.graph_data.graph_db_name}_Results_run_id_{self.run_num}_validation_step_{self.validation_num}.csv'

            # header use semicolon as delimiter
            header = ("Dataset;RunNumber;ValidationNumber;Algorithm;TrainingSize;ValidationSize;TestSize"
                      ";HyperparameterSVC;HyperparameterAlgo;ValidationAccuracy;TestAccuracy\n")

            # Save file for results and add header if the file is new
            with open(f'Results/{file_name}', "a") as file_obj:
                if os.stat(f'Results/{file_name}').st_size == 0:
                    file_obj.write(header)

            # Save results to file
            with open(f'Results/{file_name}', "a") as file_obj:
                file_obj.write(f"{self.graph_data.graph_db_name};{self.run_num};{self.validation_num};NoGKernel;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};{c_param};{0};{val_acc};{test_acc}\n")
