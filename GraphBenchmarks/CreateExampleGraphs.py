'''
Created on 26.09.2019

@author: florian
'''

import networkx as nx
import numpy as np

def example_graph1():
    G = nx.Graph()
    G.add_node(0, label = np.array([0]))
    G.add_node(1, label = np.array([0]))
    G.add_node(2, label = np.array([1]))
    G.add_node(3, label = np.array([1]))
    G.add_node(4, label = np.array([0]))
    G.add_node(5, label = np.array([0]))
    
    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])
    return G

def example_graph2():
    G = nx.Graph()
    G.add_node(0, label = np.array([0]))
    G.add_node(1, label = np.array([0]))
    G.add_node(2, label = np.array([1]))
    G.add_node(3, label = np.array([0]))
    G.add_node(4, label = np.array([0]))
    G.add_node(5, label = np.array([0]))
    
    G.add_edges_from([(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)])
    return G

def circle_graph(n = 100):
    G = nx.Graph()
    
    for i in range(0, n):
        G.add_node(i, label = np.array([i % 2]), abc = i % 2)
    
    for i in range(0, n):
        G.add_edge(i % n,(i + 1) % n)
    return G

def double_circle(n = 50, m = 50):
    G = circle_graph(n)
    for i in range(n, n + m):
        G.add_node(i, label = np.array([i % 2]), abc = i % 2)
    
    for i in range(n, n + m):
        G.add_edge(i % m, (i+1) % m)
    return G
