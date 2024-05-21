import ReadWriteGraphs.GraphDataToGraphList as rg


def get_graph_label_hist(graph):
    label_hist = {}
    for n in range(0, graph.number_of_nodes()):
        label = int(rg.node_label_vector(graph, n))
        if label not in label_hist:
            label_hist[label] = 1
        else:
            label_hist[label] += 1
    return label_hist


def draw_graphs(path="../GraphData/DS_all/", db="MUTAG", labels=None):
    gen = rg.graph_data_generator(path, db)
    while True:
        graph_data = next(gen)
        print(type(graph_data))
        graph, graph_label, graph_attribute = graph_data

        graph.add_node(graph.number_of_nodes(), label= [graph_label])
        rg.draw_graph(graph)
        print(rg.nodes_label_matrix(graph))
        print(rg.node_label_vector(graph, 0))
        print(rg.node_attribute_vector(graph, 0))
        print(type(rg.nodes_label_matrix(graph)))
        # print(nodes_label_coding_matrix(graph, 50, False))

        print(rg.edges_attribute_matrix(graph))
        print(rg.edges_label_coding_matrix(graph, 5, False))
        print(rg.has_node_labels(graph))
        print(rg.node_attribute_dimension(graph))

def draw_graph(graph, label=2):

        graph.add_node(graph.number_of_nodes(), label= [label])
        rg.draw_graph(graph)
        print(rg.nodes_label_matrix(graph))
        print(rg.node_label_vector(graph, 0))
        print(rg.node_attribute_vector(graph, 0))
        print(type(rg.nodes_label_matrix(graph)))
        # print(nodes_label_coding_matrix(graph, 50, False))

        print(rg.edges_attribute_matrix(graph))
        print(rg.edges_label_coding_matrix(graph, 5, False))
        print(rg.has_node_labels(graph))
        print(rg.node_attribute_dimension(graph))
