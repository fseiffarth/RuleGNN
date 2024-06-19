'''
Created on 27.05.2019

@author:
'''
import time

import numpy as np
import torch
import torch.nn as nn


def main():
    x = np.zeros((3, 3))
    y = np.arange(9)

    z = np.array([[0, 0],[0, 1]])
    a = np.array([0, 1])
    b = np.array([0, 0])
    c = np.array([1, 5])

    np.put(x, z, y[c])

    x = torch.zeros((3, 3))
    y = torch.arange(9)

    z = torch.tensor([0, 1])
    a = torch.tensor([0, 1])
    b = torch.tensor([0, 0])
    c = torch.tensor([1, 5])

    src = torch.tensor([[4, 3, 5],
                        [6, 7, 8]])
    src.put_(torch.tensor([1, 3]), z[a])

    print("src:", src)

    print(x, y, z)

    print(x, y, z)

    src = torch.zeros((100, 100))
    indices = torch.arange(500)
    values = torch.arange(500, dtype=torch.float32)

    idx = torch.randperm(values.nelement())
    values = values.view(-1)[idx].view(values.size())

    Param_b = nn.Parameter(torch.arange((10), dtype=torch.double))
    print(Param_b[torch.tensor([0, 1])])
    print(values)


    start = time.time()
    for i, x in enumerate(indices, 0):
        src.flatten()[x] = values[i]
    end = time.time() - start
    print(end)
    print(src)

    start = time.time()
    src.put_(indices, values)
    end = time.time() - start
    print(end)
    print(src)



    """
    gen = graph_data_generator(path, db)
    while True:
        graph_data = next(gen)
        print(type(graph_data))
        graph, graph_label, graph_attribute = graph_data
        
        draw_graph(graph)
        print(nodes_label_matrix(graph))
        print(node_label_vector(graph, 0))
        print(node_attribute_vector(graph, 0))
        print(type(nodes_label_matrix(graph)))
        #print(nodes_label_coding_matrix(graph, 50, False))
        
        print(edges_attribute_matrix(graph))
        print(edges_label_coding_matrix(graph, 5, False))
        print(has_node_labels(graph))
        print(node_attribute_dimension(graph))
        """

if __name__ == '__main__':
    main()