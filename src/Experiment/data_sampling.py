from typing import List

import numpy as np

from src.utils.GraphData import ShareGNNDataset


def curriculum_sampling_graph_size(graph_data: ShareGNNDataset, training_data:np.ndarray, bucket_num:int, num_batches:int, batch_size:int, total_epochs:int, epoch:int, anti:bool=False)->np.ndarray:
    """
    This function is used to get the graph size for the curriculum learning.
    :param training_data:
    :param num_batches:
    :param batch_size:
    :param epoch:
    :param anti:
    :param total_epochs:
    :param bucket_num: The number of buckets for the curriculum learning
    :param graph_data: The graph data
    :return: The graph size for the curriculum learning
    """
    # get the graph size for the curriculum learning
    training_graph_sizes = graph_data.slices['x'][training_data + 1] - graph_data.slices['x'][training_data]
    # sort the graphs based on the graph size (togehter with the index)
    training_graph_sizes, indices = np.sort(training_graph_sizes), np.argsort(training_graph_sizes)
    # devide the graph indices into bucket_num buckets
    index_buckets = np.array_split(indices, bucket_num)
    # get current bucket from current epoch
    current_bucket_index = int((epoch / total_epochs) * bucket_num)
    if anti:
        current_bucket_index = bucket_num - 1 - current_bucket_index
    # get the current bucket
    current_bucket = index_buckets[current_bucket_index]
    # sample num_batches batches from the current bucket
    training_samples = np.zeros((num_batches, batch_size), dtype=int)
    for i in range(num_batches):
        training_samples[i] = np.random.choice(current_bucket, batch_size, replace=True)
    return training_samples

def no_curriculum_sampling(training_data:np.ndarray, num_batches:int, batch_size:int)->np.ndarray:
    """
    This function is used to get the graph size for the no curriculum learning.
    :param training_data:
    :param num_batches:
    :param batch_size:
    :return: The graph size for the no curriculum learning
    """
    # sample num_batches batches from the training data
    training_samples = np.zeros((num_batches, batch_size))
    for i in range(num_batches):
        training_samples[i] = np.random.choice(training_data, batch_size, replace=True)
    return training_samples