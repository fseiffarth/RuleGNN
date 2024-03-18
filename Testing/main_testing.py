import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import LambdaLR
import random
import TrainTestData.TrainTestData as ttd
from Parameters import Parameters

import itertools
from NeuralNetArchitectures import GraphNN
from Time.TimeClass import TimeClass

def main():
    # get gpu or cpu: not used at the moment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # path do db and db
    data_path = "../DS_all/"
    results_path = "../Results/Test"
    db = "MUTAG"
    # db = "PTC_MR"

    timer = TimeClass()

    para = Parameters.Parameters()

    """
        Data parameters
    """
    para.set_data_param(path=data_path, results_path=results_path,
                        db=db,
                        max_coding=1, batch_size=64, node_features=1, node_labels=7, edge_labels=4)

    """
        Network parameters
    """
    para.set_evaluation_param(n_runs=1, n_val_runs=10, n_epochs=100, balance_data=False)
    """
        Network hyper parameters
        """
    # para.set_evaluation_param(n_runs = 10, n_val_runs = 10, n_epoches = 200, balance_data = False)
    """
        Print, save and draw parameters
        """
    para.set_print_param(net_print_weights=True, print_number=1, draw=False)

    """
        Create Input data, information and targets from the graphs for training and testing
        """
    distance_list = []
    cycle_list = []
    #Data, Labels, graph_data = test_graphdata.four_nodes(distance_list, cycle_list)
    Data, Labels, graph_data = ttd.data_from_graph_db(para.path, para.db, distance_list, cycle_list,
                                                      labels_zero_one=True)
    print(Labels)

    graph_number = len(Labels)
    zero_one = True

    if para.draw:
        lines, coord = ttd.plot_init(2)

    """Save File"""
    para.set_save_file()

    accuracies = []

    epoch_accuracies = []
    test_accuracies = []

    # training

    for run in range(0, para.n_runs):
        run_test_accuracies = []
        run_epoch_accuracies = []
        """
            Split the data in training validation and test set
            """
        seed_list = [1, 13, 5, 5, 687348, 374534, 35675, 3567, 3486467, 488467, 364, 90, 6, 32, 46356537, 47887]
        training_data, validate_data, test_data = ttd.get_train_validation_test_list(graph_number, seed=seed_list[run],
                                                                                     balanced=False,
                                                                                     graph_labels=Labels,
                                                                                     test_divide=1 / para.n_val_runs,
                                                                                     val_divide=0)
        epoch_accuracy = 0

        for k_val in range(0, para.n_val_runs):
            """
                Set up the network
                """
            net = GraphNN.GraphTestNet(graph_data=graph_data, n_node_features=para.node_features,
                                   n_node_labels=para.node_labels, n_edge_labels=para.edge_labels,
                                   distance_list=distance_list, cycle_list=cycle_list,
                                   print_weights=para.net_print_weights)
            # net = nn.DataParallel(net)
            optimizer = optim.Adam(net.parameters(), lr=0.1)
            criterion = nn.MSELoss()
            if run == 0 and k_val == 0:
                para.save_net_params(net)

            """
                Variable learning rate
                """
            scheduler_on = False
            if scheduler_on:
                lambda1 = lambda epoch: 0.1 / ((10 * epoch) / 50 + 1)
                scheduler = LambdaLR(optimizer, lr_lambda=lambda1)

            """
                Create balanced training data
                """
            if para.balance_data:
                training_balanced = ttd.balance_data(training_data, Labels)
            else:
                training_balanced = training_data.copy()
            para.save_train_test(training_balanced, test_data)

            """
                Run through the defined number of epochs
                """
            for epoch in range(para.n_epochs):
                timer.measure("epoch")
                net.epoch = epoch
                epoch_loss = 0.0
                running_loss = 0.0
                epoch_acc = 0.0

                """
                    Random Train batches for each epoch
                    """
                t_shuffle = training_balanced.copy()
                random.shuffle(t_shuffle)
                train_batches = ttd.get_training_batch(t_shuffle, para.batch_size)

                """
                    for param_group in optimizer.param_groups:
                        print(param_group['lr'])
                    """
                if scheduler_on:
                    scheduler.step()

                for i, batch in enumerate(train_batches, 0):
                    timer.measure("forward")
                    optimizer.zero_grad()
                    outputs = Variable(torch.zeros((len(batch), 1), dtype=torch.double))
                    targets = Variable(torch.zeros((len(batch), 1), dtype=torch.double))

                    # parallelize forward
                    for j, batch_pos in enumerate(batch, 0):
                        targets[j][0] = Labels[batch_pos]
                        net.train(True)
                        timer.measure("forward_step")
                        outputs[j][0] = net(Variable(Data[batch_pos]).to(device), batch_pos)
                        timer.measure("forward_step")

                    loss = criterion(outputs, targets)
                    timer.measure("forward")

                    timer.measure("backward")
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    #print(net.l1.name, [x.item() for x in net.l1.Param_W])
                    #print(net.l2.name, [x.item() for x in net.l2.Param_W])
                    #print(net.lr.name, [x.item() for x in net.lr.Param_W])
                    timer.measure("backward")

                    timer.reset()

                    running_loss += loss.item()
                    epoch_loss += running_loss
                    if epoch % para.print_number == 0:
                        epoch_acc += 100 * ttd.get_accuracy(outputs, targets, zero_one=zero_one) * (
                                    len(batch) / len(training_balanced))
                        print("\tepoch: {}/{}, batch: {}/{}, loss: {}, acc: {} % ".format(epoch + 1, para.n_epochs,
                                                                                          i + 1, len(train_batches),
                                                                                          running_loss,
                                                                                          100 * ttd.get_accuracy(
                                                                                              outputs, targets,
                                                                                              zero_one=zero_one)))
                        para.count += 1
                        running_loss = 0.0

                test_acc = "Not evaluated"
                # test accuracy
                if epoch % para.print_number == 0:
                    test_acc = 0
                    outputs = Variable(torch.zeros((len(test_data), 1), dtype=torch.double))
                    targets = Variable(torch.zeros((len(test_data), 1), dtype=torch.double))
                    for j, data_pos in enumerate(test_data, 0):
                        inputs = Variable(torch.DoubleTensor(Data[data_pos]))
                        targets[j][0] = Labels[data_pos]
                        net.train(False)
                        outputs[j][0] = net(inputs, data_pos)[0]
                    test_acc = 100 * ttd.get_accuracy(outputs, targets, zero_one=zero_one)
                    print("test acc: {}".format(test_acc))

                timer.measure("epoch")
                # epoch accuracy
                epoch_time = timer.get_flag_time("epoch")
                if epoch % para.print_number == 0:

                    print("run: {} val step: {} epoch loss: {} epoch acc: {} time: {}".format(run, k_val, epoch_loss,
                                                                                              epoch_acc,
                                                                                              epoch_time))

                if para.draw:
                    # print some statistics
                    coord = ttd.add_values([epoch + 1, epoch + 1], [epoch_acc, test_acc], coord)
                    lines = ttd.live_plotter_lines(coord, lines)

                if epoch in [0, 49, 99, 149, 199, 249, 299, 341, 399]:
                    # Save file for results
                    para.save_run(str(run), str(k_val), str(epoch), epoch_time, str(epoch_loss), str(epoch_acc), str(test_acc))
                epoch_accuracy = epoch_acc

            """Evaluation"""
            outputs = Variable(torch.zeros((len(test_data), 1), dtype=torch.double))
            targets = Variable(torch.zeros((len(test_data), 1), dtype=torch.double))
            for j, data_pos in enumerate(test_data, 0):
                inputs = Variable(torch.DoubleTensor(Data[data_pos]))
                targets[j][0] = Labels[data_pos]
                net.train(False)
                outputs[j][0] = net(inputs, data_pos)[0]
            test_acc = 100 * ttd.get_accuracy(outputs, targets, zero_one=zero_one)

            accuracies.append(test_acc)
            print(outputs, targets)

            """Set new training and test for next cross validation step"""
            training_data = np.zeros(graph_number, dtype=np.int16)
            itertool_counter = 0
            for item in itertools.chain(training_balanced, test_data):
                training_data[itertool_counter] = item
                itertool_counter += 1
            test_size_list = [graph_number // int(para.n_val_runs) + (1 if x < graph_number % int(para.n_val_runs) else 0) for x in
                              range(para.n_val_runs)]

            if k_val < para.n_val_runs - 1:
                test_data = training_data[:test_size_list[k_val+1]]
                training_data = training_data[test_size_list[k_val+1]:]

            # Save results for one validation run
            para.save_predictions(outputs, targets)
            para.save_result(run, k_val, str(epoch_accuracy), str(test_acc))
            test_accuracies.append(test_acc)
            run_test_accuracies.append(test_acc)
            run_epoch_accuracies.append(epoch_accuracy)
            epoch_accuracies.append(epoch_accuracy)

        para.save_average_result(run, run_epoch_accuracies, run_test_accuracies)

    para.save_average_result(run, epoch_accuracies, test_accuracies)

if __name__ == '__main__':
    main()