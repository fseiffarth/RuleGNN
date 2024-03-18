# GraphDataToNetworkxGraphList

**Reads graph data from:**

  [https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets)

and outputs them as a list of python networkx graphs, see the details below.

**Functions:**

1. **graph_data_to_graph_list(path, db)**  *Takes database path {path} and database-name {db} and outputs the graph data in form of a triple: (graph_list, graph_labels, graph_attributes)*

     -graph_list:   *#python list of networkx graphs with graph information given by the database*
  
     -graph_labels:   *#python list of integers as the labels of the graphs according to the database*
  
     -graph_attributes:   *#python list with additional attributes of the graphs according to the database or empty list if there are none*

2. **node_label_vector(graph, node_id)**  *#outputs vector of node_label of node with {node_id} in graph {graph}*

3. **node_attribute_vector(graph, node_id)**  *#outputs vector of node_attributes of node with {node_id} in graph {graph}*
  
  
 
4. **nodes_label_matrix(graph)**  *#same as above but outputs all labels of nodes of graph {graph}*

5. **nodes_attribute_matrix(graph)**  *#same as above but outputs all attributes of nodes of graph {graph}*

  
  
6. **edge_label(graph, node_i, node_j)**  *#outputs vector of edge_labels of edge (node_i, node_j) in graph {graph}*

7. **edge_attribute_matrix(graph, node_i, node_j)**   *#outputs vector of edge_attributes of edge (node_i, node_j) in graph {graph}*

  





=======
**Python example:**
```
path = "path_to_dbs/"
db = "MUTAG"
graph_data = graph_data_to_graph_list(path, db)
graph_list, graph_labels, graph_attributes = graph_data
```
