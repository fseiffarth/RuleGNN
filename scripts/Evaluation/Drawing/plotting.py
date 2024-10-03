import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx.algorithms.bipartite.basic import color

from src.Architectures.RuleGNN.RuleGNNLayers import RuleConvolutionLayer


def rules_vs_occurences(layer:RuleConvolutionLayer) -> np.ndarray:
    weight_distribution = layer.weight_distribution
    num_weights = layer.Param_W.shape[0]
    weight_array = np.zeros(num_weights)
    for i, weights in enumerate(weight_distribution):
        weight_pos = weights[:, 3]
        # in weight_array add 1 where the index is in weight_pos
        weight_array[weight_pos] += 1
    # sort the weight_array (largest occurence first) and save the sorted indices
    sort_indices = np.argsort(weight_array)[::-1]
    weight_array = weight_array[sort_indices]
    # get the number of non-zero elements
    num_non_zero = np.count_nonzero(weight_array)
    print(f'Number of non-zero elements: {num_non_zero}')
    # plot the distribution of the rules as bar plot
    fig, ax = plt.subplots()
    # set size of the
    ax.scatter(np.arange(num_weights), weight_array, s=0.5)
    plt.xlabel('Rule index')
    plt.ylabel('Occurences')
    plt.title('Distribution of rules')
    plt.show()
    return sort_indices

def rules_vs_occurences_properties(layer:RuleConvolutionLayer):
    weight_distribution = layer.weight_distribution
    num_weights = layer.Param_W.shape[0]
    weight_array = np.zeros(num_weights)
    for i, weights in enumerate(weight_distribution):
        weight_pos = weights[:, 3]
        # in weight_array add 1 where the index is in weight_pos
        weight_array[weight_pos] += 1
    # define layer.n_properties colors for the properties from an existing colormap
    prop_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    # convert to hex colors
    # color weights based on properties
    weight_colors = []
    for i, p in enumerate(range(layer.n_properties)):
        weight_colors += [prop_colors[i] for _ in range(num_weights//layer.n_properties)]
    weight_colors = np.array(weight_colors)
    # plot the distribution of the rules
    fig, ax = plt.subplots()
    plt.figure()
    ax.bar(np.arange(num_weights), weight_array, color=weight_colors)
    plt.xlabel('Rule index')
    plt.ylabel('Occurences')
    plt.title('Distribution of rules')
    plt.show()

def rules_vs_weights(layer:RuleConvolutionLayer, sort_indices:np.ndarray):
    weights = layer.Param_W.detach().cpu().numpy()
    weights = weights[sort_indices]
    # plot the distribution of the rules
    fig, ax = plt.subplots()
    ax.scatter(np.arange(len(weights)), weights, s=0.5)
    plt.xlabel('Rule index')
    plt.ylabel('Occurences')
    plt.title('Distribution of rules')
    plt.show()
