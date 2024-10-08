'''
Created on 14.03.2019

@author:
'''
import random
import matplotlib.pyplot as plt
import src.utils.ReadWriteGraphs.GraphDataToGraphList as gdtgl
import numpy as np
import torch

def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def get_train_test_list(size, divide=0.9, seed=10):
    random.seed(seed)
    data = [i for i in range(0, size)]
    train_data = random.sample(data, int(len(data) * divide))
    test_data = diff(data, train_data)
    return train_data, test_data


def get_train_validation_test_list(test_indices, validation_step, seed=10, balanced=False, graph_labels=[], val_size=0):
    test_data = test_indices[validation_step]
    train_data_unb = np.concatenate([x for i, x in enumerate(test_indices) if i != validation_step])
    np.random.seed(seed)
    np.random.shuffle(train_data_unb)
    val_data = []
    if val_size > 0:
        val_data, train_data_unb = np.split(train_data_unb, [int(val_size * train_data_unb.size)])
    # sort validation data
    val_data = np.sort(val_data)
    train_data_b = train_data_unb.copy()

    # create balanced training set
    if balanced and graph_labels:
        label_dict = {}
        max_class = 0

        # get all classes of labels by name
        for x in train_data_unb:
            if str(graph_labels[x]) in label_dict.keys():
                label_dict[str(graph_labels[x])] += 1
            else:
                label = str(graph_labels[x])
                label_dict[label] = 1

        # get biggest class of labels
        for x in label_dict.keys():
            if label_dict[x] > max_class:
                max_class = label_dict[x]

        # dublicate samples from smaller classes
        for x in label_dict.keys():
            if label_dict[x] < max_class:
                number = max_class - label_dict[x]
                # get all indices of train_data_unb with graph_labels == x
                train_data_class_x = [i for i in train_data_b if graph_labels[i] == int(x)]
                # get new_array size random values from train_data_class_x
                np.random.seed(seed)
                new_array = np.random.choice(train_data_class_x, number, replace=True)
                # add the new array to the training data
                train_data_b = np.append(train_data_b, new_array)
        # sort training data
        train_data_b = np.sort(train_data_b)
        return np.asarray(train_data_b), np.asarray(val_data, dtype=int), test_data
    else:
        # sort training data
        train_data_unb = np.sort(train_data_unb)
        return train_data_unb, np.asarray(val_data, dtype=int), test_data


def balance_data(data, labels):
    train_data_b = data.copy()
    label_dict = {}
    max_class = 0

    # get all classes of labels by name
    for x in data:
        if str(labels[x]) in label_dict.keys():
            label_dict[str(labels[x])] += 1
        else:
            label = str(labels[x])
            label_dict[label] = 1

    # get biggest class of labels
    for x in label_dict.keys():
        if label_dict[x] > max_class:
            max_class = label_dict[x]

    # dublicate samples from smaller classes
    for x in label_dict.keys():
        if label_dict[x] < max_class:
            number = max_class - label_dict[x]
            counter = 0
            while counter < number:
                for t in data:
                    if counter < number and str(labels[t]) == x:
                        train_data_b.append(t)
                        counter += 1
    return train_data_b


def get_training_batch(training_data, batch_size):
    data = [training_data[x:min(x + batch_size, len(training_data))] for x in range(0, len(training_data), batch_size)]
    return data


def get_accuracy(output, labels, one_hot_encoding=True, zero_one=False):
    counter = 0
    correct = 0
    if one_hot_encoding:
        for i, x in enumerate(output, 0):
            if torch.argmax(x) == torch.argmax(labels[i]):
                correct += 1
            counter += 1
    else:
        if zero_one:
            for i, x in enumerate(output, 0):
                if abs(x - labels[i]) < 0.5:
                    correct += 1
                counter += 1
        else:
            for i, x in enumerate(output, 0):
                if x * labels[i] > 0:
                    correct += 1
                counter += 1
    return correct / counter


def live_plotter(x_vec, y1_data, line1, identifier='', pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec, y1_data, '-o', alpha=0.8)
        # update plot label/title
        plt.ylabel('Y Label')
        plt.title('Title: {}'.format(identifier))
        plt.show()

    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    if np.min(y1_data) <= line1.axes.get_ylim()[0] or np.max(y1_data) >= line1.axes.get_ylim()[1]:
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1


def plot_init(line_num, identifier='', epochs=100):
    coordinates = [[], []]
    lines = []
    # this is the call to matplotlib that allows dynamic plotting
    plt.ion()
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111)
    # update plot label/title
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    # set y range min and max
    plt.ylim([0, 1])
    # set x range min and max
    plt.xlim([1, epochs])

    plt.title('Title: {}'.format(identifier))
    plt.show()

    for i in range(0, line_num):
        # create a variable for the line so we can later update it
        line, = ax.plot(0, 0, '-o', alpha=0.8)
        lines.append(line)
        coordinates[0].append(np.zeros(1))
        coordinates[1].append(np.zeros(1))
    return lines, coordinates


def plot_learning_data(new_x, new_y, data, epochs, title=''):
    plt.ion()
    plt.clf()

    # set y range min and max
    plt.ylim([0, 100])
    # set x range min and max
    plt.xlim([1, new_x])

    plt.title(f"Title: {title}")
    # add new_x new_y to data which is a map of lists
    for i, y in enumerate(new_y, 0):
        if len(data['Y']) <= i:
            data['Y'].append([])
            data['X'].append([])
        data['Y'][i].append(y)
        data['X'][i].append(new_x)
    # plot data as multiple lines with different colors in one plot
    for i, x in enumerate(data['X'], 0):
        plt.plot(x, data['Y'][i])

    # add legend to the lines epoch accuracy, validation accuracy and test accuracy
    plt.legend(['Train', 'Validation', 'Test', 'Loss'], loc='lower right')

    plt.draw()
    plt.pause(0.001)
    return data


def add_values(xvalues, yvalues, coordinates):
    for i, x in enumerate(xvalues, 0):
        coordinates[0][i] = np.append(coordinates[0][i], x)
        coordinates[1][i] = np.append(coordinates[1][i], yvalues[i])

    return coordinates


def live_plotter_lines(coordinates, lines):
    for i, line in enumerate(lines, 0):
        # after the figure, axis, and line are created, we only need to update the y-data
        line.set_ydata(coordinates[1][i])
        line.set_xdata(coordinates[0][i])
    # return line so we can update it again in the next iteration
    return lines


def get_data_indices(size, seed, kFold):
    random.seed(seed)
    np.random.seed(seed)
    data = np.arange(0, size)
    np.random.shuffle(data)
    data = np.array_split(data, kFold)
    # sort the data
    for i in range(0, len(data)):
        data[i] = np.sort(data[i])
    return data
