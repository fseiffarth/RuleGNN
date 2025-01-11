import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx.algorithms.bipartite.basic import color

from src.Architectures.RuleGNN.RuleGNNLayers import RuleConvolutionLayer

def rules_vs_occurences(layer: RuleConvolutionLayer, db_name, channel=0) -> np.ndarray:
    weight_distribution = layer.weight_distribution
    num_weights = layer.Param_W.shape[0]
    weight_array = np.zeros(num_weights)
    weights = weight_distribution[:, 3]
    # get unique counts of entries in weights
    weight_array = np.bincount(weights)
    # sort the weight_array (largest occurence first) and save the sorted indices
    sort_indices = np.argsort(weight_array)[::-1]

    weight_array = weight_array[sort_indices]


    # get array such that in entry i is the index of weight_array where value is the first time larger than i
    # iterate over weight_array in reverse order
    steps = np.zeros(11)
    current_step = 1
    for i in range(num_weights-1, 0, -1):
        if weight_array[i] != current_step:
            steps[current_step] = i - 1
            current_step += 1
            if current_step == 11:
                break

    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'{layer.property_names[channel]} {i}' for i in range(layer.n_properties[channel])]
    f = lambda x : np.max(np.where(x >= np.array(layer.skips)))
    f_vectorized = np.vectorize(f)
    # get property id from sort indices using the skips
    property_indices = f_vectorized(sort_indices)

    node_colors = np.array(property_colors)[property_indices]

    # plot the distribution of the rules with legend
    fig, ax = plt.subplots()
    for i, p in enumerate(range(layer.n_properties[channel])):
        ax.scatter([], [], color=property_colors[i], label=property_legend[i])
    ax.scatter(np.arange(num_weights), weight_array, s=0.5, alpha=1, c=node_colors)
    # add legend title

    # add vertical lines for the steps
    for i in range(1, 11):
        ax.axvline(steps[i], color='black', linestyle='--', linewidth=0.5)

    ax.legend()
    plt.xlabel('Learnable parameters')
    plt.ylabel('\# Occurrences in Dataset')
    #plt.title('Number of occurrences per rule')
    # use pgf backend for latex
    plt.savefig(f'scripts/Evaluation/Drawing/Figures/occurrences_per_rule_{db_name}.png')
    plt.show()
    return sort_indices, steps

def rules_vs_weights(layer:RuleConvolutionLayer, sort_indices:np.ndarray, steps,db_name, channel=0):
    weights = layer.Param_W.detach().cpu().numpy()
    weights = weights[sort_indices]
    # colors from tab20
    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'{layer.property_names[channel]} {i}' for i in range(layer.n_properties[channel])]
    f = lambda x : np.max(np.where(x >= np.array(layer.skips)))
    f_vectorized = np.vectorize(f)
    # get property id from sort indices using the skips
    property_indices = f_vectorized(sort_indices)

    node_colors = np.array(property_colors)[property_indices]

    # plot the distribution of the rules with legend
    fig, ax = plt.subplots()
    for i, p in enumerate(range(layer.n_properties[channel])):
        ax.scatter([], [], color=property_colors[i], label=property_legend[i])

    # add vertical lines for the steps
    for i in range(1, 11):
        ax.axvline(steps[i], color='black', linestyle='--', linewidth=0.5)

    ax.scatter(np.arange(len(weights)), weights, s=1, alpha=1, c=node_colors)
    ax.legend()
    plt.xlabel('Learnable parameters (sorted by occurrences)')
    plt.ylabel('Value of the parameter')
    #plt.title('Distribution of rules')
    plt.savefig(f'scripts/Evaluation/Drawing/Figures/weights_per_rule_{db_name}.png')
    plt.show()
