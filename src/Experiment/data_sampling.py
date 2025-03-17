from typing import List

import numpy as np
import torch

from src.utils.GraphData import ShareGNNDataset


def curriculum_sampling_graph_size(graph_data: ShareGNNDataset,
                                   training_data:np.ndarray,
                                   bucket_num:int,
                                   num_batches:int,
                                   batch_size:int,
                                   total_epochs:int,
                                   epoch:int,
                                   anti:bool=False,
                                   exclusive=True,
                                   use_edges=False)->np.ndarray:
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
    :param exclusive: If True, the buckets are exclusive, otherwise they are not exclusive
    :return: The graph size for the curriculum learning
    """
    # get the graph size for the curriculum learning
    if use_edges:
        training_graph_sizes = graph_data.slices['edge_index'][training_data + 1] - graph_data.slices['edge_index'][training_data]
    else:
        training_graph_sizes = graph_data.slices['x'][training_data + 1] - graph_data.slices['x'][training_data]
    index_with_graph_size = torch.zeros((len(training_graph_sizes), 2), dtype=torch.int64)
    index_with_graph_size[:, 0] = torch.arange(len(training_graph_sizes))
    index_with_graph_size[:, 1] = training_graph_sizes
    # sort the graphs based on the graph size
    index_with_graph_size = index_with_graph_size[index_with_graph_size[:, 1].argsort()]
    indices = index_with_graph_size[:, 0].numpy()
    # devide the graph indices into bucket_num buckets
    if exclusive:
        index_buckets = np.array_split(indices, bucket_num)
    else:
        # split the array into bucket_num buckets and then add all buckets smaller than the current bucket to the current bucket
        index_buckets = np.array_split(indices, bucket_num)
        for i in range(1, bucket_num):
            index_buckets[i] = np.concatenate((index_buckets[i], index_buckets[i - 1]))


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

def curriculum_sampling_edge_num(graph_data: ShareGNNDataset,
                                    training_data:np.ndarray,
                                    bucket_num:int,
                                    num_batches:int,
                                    batch_size:int,
                                    total_epochs:int,
                                    epoch:int,
                                    anti:bool=False,
                                    exclusive=True)->np.ndarray:
    return curriculum_sampling_graph_size(graph_data=graph_data,
                                          training_data=training_data,
                                           bucket_num=bucket_num,
                                           num_batches=num_batches,
                                           batch_size=batch_size,
                                           total_epochs=total_epochs,
                                           epoch=epoch,
                                           anti=anti,
                                           exclusive=exclusive,
                                           use_edges=True)


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