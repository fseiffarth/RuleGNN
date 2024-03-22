#
import os
from typing import List
import numpy as np
from grakel import WeisfeilerLehman, VertexHistogram
from sklearn.svm import SVC

from Competitors.nx_to_grakel import nx_to_grakel
from GraphData.GraphData import GraphData
from sklearn.metrics import accuracy_score


class WLKernel:
    def __init__(self, graph_data: GraphData, run_num: int, validation_num: int, training_data: List[int],
                 validate_data: List[int], test_data: List[int],
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

        train_graphs = [self.graph_data.graphs[i] for i in self.training_data]
        val_graphs = [self.graph_data.graphs[i] for i in self.validate_data]
        test_graphs = [self.graph_data.graphs[i] for i in self.test_data]

        y_train = np.asarray([self.graph_data.graph_labels[i] for i in self.training_data])
        y_val = np.asarray([self.graph_data.graph_labels[i] for i in self.validate_data])
        y_test = np.asarray([self.graph_data.graph_labels[i] for i in self.test_data])

        grakel_train = nx_to_grakel(train_graphs)
        grakel_val = nx_to_grakel(val_graphs)
        grakel_test = nx_to_grakel(test_graphs)

        for n_iter in range(1, 15):
            for c_param in [-11, -9, -7, -5, -3, -1, 0, 1, 3, 5, 7, 9, 11]:
                c_param = 2 ** c_param
                gk = WeisfeilerLehman(n_iter=n_iter, base_graph_kernel=VertexHistogram, normalize=True)
                K_train = gk.fit_transform(grakel_train)
                K_val = gk.transform(grakel_val)
                K_test = gk.transform(grakel_test)

                # Uses the SVM classifier to perform classification
                clf = SVC(C=c_param, kernel="precomputed", random_state=self.seed)
                clf.fit(K_train, y_train)
                y_val_pred = clf.predict(K_val)
                y_test_pred = clf.predict(K_test)

                # compute the validation accuracy and test accuracy and print it
                val_acc = accuracy_score(y_val, y_val_pred)
                test_acc = accuracy_score(y_test, y_test_pred)

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
                    file_obj.write(
                        f"{self.graph_data.graph_db_name};{self.run_num};{self.validation_num};WLKernel;{len(self.training_data)};{len(self.validate_data)};{len(self.test_data)};{c_param};{n_iter};{val_acc};{test_acc}\n")
