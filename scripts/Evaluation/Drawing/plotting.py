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

    weights_per_property = int(layer.weight_num/layer.n_properties)
    # invert layer.non_zero_weight_map
    non_zero_weight_map = {v: k for k, v in layer.non_zero_weight_map.items()}
    # colors from tab20

    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'Distance {i+1}' for i in range(layer.n_properties)]
    node_colors = []
    for i, _ in enumerate(weight_array):
        idx = sort_indices[i]
        non_zero_idx = non_zero_weight_map[idx]
        node_colors.append(property_colors[non_zero_idx//weights_per_property])

    # plot the distribution of the rules with legend
    fig, ax = plt.subplots()
    for i, p in enumerate(range(layer.n_properties)):
        ax.scatter([], [], c=property_colors[i], label=property_legend[i])
    ax.scatter(np.arange(num_weights), weight_array, s=0.5, alpha=1, c=node_colors)
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
    property_colors = plt.get_cmap('tab20').colors
    # color weights based on properties
    weight_colors = []
    for i, p in enumerate(range(layer.n_properties)):
        weight_colors += [property_colors[i] for _ in range(num_weights//layer.n_properties)]
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

    weights_per_property = int(layer.weight_num/layer.n_properties)
    # invert layer.non_zero_weight_map
    non_zero_weight_map = {v: k for k, v in layer.non_zero_weight_map.items()}
    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'Distance {i+1}' for i in range(layer.n_properties)]
    node_colors = []
    for i, _ in enumerate(weights):
        node_colors.append(property_colors[non_zero_weight_map[sort_indices[i]]//weights_per_property])

    # plot the distribution of the rules with legend
    fig, ax = plt.subplots()
    for i, p in enumerate(range(layer.n_properties)):
        ax.scatter([], [], c=property_colors[i], label=property_legend[i])

    ax.scatter(np.arange(len(weights)), weights, s=1, alpha=1, c=node_colors)
    ax.legend()
    plt.xlabel('Rule index')
    plt.ylabel('Occurences')
    plt.title('Distribution of rules')
    plt.show()
