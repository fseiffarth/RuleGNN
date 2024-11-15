import numpy as np
import torch
from matplotlib import pyplot as plt
from networkx.algorithms.bipartite.basic import color

from src.Architectures.RuleGNN.RuleGNNLayers import RuleConvolutionLayer


def rules_vs_occurences(layer: RuleConvolutionLayer, db_name, channel=0) -> np.ndarray:
    weight_distribution = layer.weight_distribution
    num_weights = layer.Param_W.shape[0]
    weight_array = np.zeros(num_weights)
    for i, weights in enumerate(weight_distribution):
        weight_pos = weights[:, 3]
        # in weight_array add 1 where the index is in weight_pos
        for pos in weight_pos:
            weight_array[pos] += 1
    # sort the weight_array (largest occurence first) and save the sorted indices
    sort_indices = np.argsort(weight_array)[::-1]
    # invert layer.non_zero_weight_map
    threshold_idx_map = {new_idx: old_idx for old_idx, new_idx in enumerate(layer.threshold_weight_map) if new_idx != -1}
    # colors from tab20
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








    weights_per_property = int(np.sum(layer.weight_num)/layer.n_properties[channel])
    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'Distance {i}' for i in range(layer.n_properties[channel])]
    node_colors = []
    for i, _ in enumerate(weight_array):
        new_idx = sort_indices[i]
        old_idx = threshold_idx_map[new_idx]
        node_colors.append(property_colors[old_idx//weights_per_property])

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
    plt.xlabel('Rules')
    plt.ylabel('\# Occurrences')
    #plt.title('Number of occurrences per rule')
    # use pgf backend for latex
    plt.savefig(f'scripts/Evaluation/Drawing/Figures/occurrences_per_rule_{db_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    return sort_indices, steps

def rules_vs_weights(layer:RuleConvolutionLayer, sort_indices:np.ndarray, steps,db_name, channel=0):
    weights = layer.Param_W.detach().cpu().numpy()
    weights = weights[sort_indices]

    weights_per_property = int(np.sum(layer.weight_num)/layer.n_properties[channel])
    # invert layer.non_zero_weight_map
    threshold_idx_map = {new_idx: old_idx for old_idx, new_idx in enumerate(layer.threshold_weight_map) if new_idx != -1}
    # colors from tab20
    property_colors = plt.get_cmap('tab20').colors
    property_legend = [f'Distance {i}' for i in range(layer.n_properties[channel])]
    node_colors = []
    for i, _ in enumerate(weights):
        node_colors.append(property_colors[threshold_idx_map[sort_indices[i]]//weights_per_property])

    # plot the distribution of the rules with legend
    fig, ax = plt.subplots()
    for i, p in enumerate(range(layer.n_properties[channel])):
        ax.scatter([], [], color=property_colors[i], label=property_legend[i])

    # add vertical lines for the steps
    for i in range(1, 11):
        ax.axvline(steps[i], color='black', linestyle='--', linewidth=0.5)

    ax.scatter(np.arange(len(weights)), weights, s=1, alpha=1, c=node_colors)
    ax.legend()
    plt.xlabel('Rule')
    plt.ylabel('Attention weight')
    plt.title('Distribution of rules')
    plt.savefig(f'scripts/Evaluation/Drawing/Figures/weights_per_rule_{db_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
