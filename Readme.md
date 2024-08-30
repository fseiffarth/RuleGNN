# Using RuleGNNs for Graph Classification

This repository contains an explanation on how to [reproduce](#Reproduce-RuleGNN-Experiments) the results of our paper [Rule Based Learning with Dynamic (Graph) Neural Networks](https://arxiv.org/abs/2406.09954).
Moreover, we give a detailed explanation on how to use RuleGNNs for [custom datasets](#Custom-Datasets).

# Reproduce RuleGNN Experiments
To reproduce the experiments of the paper, follow the steps below. All necessary code can be found in the [Reproduce_RuleGNN](Reproduce_RuleGNN) folder.
1. Clone the repository using
    ```bash
    git clone git@github.com:fseiffarth/RuleGNN.git
    ```
2. Install the required packages using the environment.yml file using the following command:
   ```bash
   conda env create -f environment.yml
   ```
3. Run the file [Reproduce_RuleGNN/main.py](Reproduce_RuleGNN/main.py) to reproduce the experiments of the paper. This will:
   - download the datasets
   - preprocess the datasets
   - find the best hyperparameters for different models
   - run the best models three times with different seeds
   - evaluate the results

   All results will be saved in the [Reproduce_RuleGNN/Results](Reproduce_RuleGNN/Results) folder.

# Experiments on the TU Dortmund Graph Benchmark
All datasets from the TU Dortmund Benchmark available [here](https://chrsmrrs.github.io/datasets/docs/datasets/) can be used directly for experiments as shown in [Examples/TUExample](Examples/TUExample).

# Customize Experiments
First of all your dataset needs to be in the correct format.
At the moment, the code supports two different options.

- Option 1: Add a function ```favorite_graph_dataset``` to [src/utils/SyntheticGraphs.py](src/utils/SyntheticGraphs.py) 
   that returns a tuple of the form 
    ```python
      (List[networkx.Graph], List[int/float])
   ```
   where the first list contains the graphs (optional with node and edge labels) and the second list contains the labels of the graphs. 

- Option 2 (preferred): Save your favorite graph dataset in the format described below in [Data Format](#Data-Format).

To run the experiment you only need the following code:
   ```python
   from pathlib import Path
   
   from scripts.ExperimentMain import ExperimentMain
   
   
   def main():
       experiment = ExperimentMain(Path('Path/To/Your/Main/Config/File.yml'))
       experiment.Preprocess()
       experiment.Run()
   
   if __name__ == '__main__':
       main()
   ```
In the preprocessing step, the data will be downloaded, generated and labels and properties according to the configuration file described below are precomputed.
The run step consists of first finding the best hyperparameters for the model using a 10-fold cross-validation.
The best model according to the validation set is then trained three times with different seeds and evaluated on the test set.
The results are saved as ```summary.csv``` resp. ```summary_best_model.csv``` in the results folder specified in the configuration file.


All the details are defined in two configuration files.

### Main Config File
In the main config file, you define which datasets you want to use and how to split the data into training, validation, and test sets.
The file should look like this:
```yaml
datasets:
  # in case of a given generation function called ring_diagonals in this case
  - {name: "EXAMPLE_DB", validation_folds: 10, experiment_config_file: "Examples/CustomExample/Configs/config_experiment.yml", type: "generate_from_function", generate_function: ring_diagonals, generate_function_args: {data_size: 1000, ring_size: 50}}
  # in case of a dataset from the TU Dortmund Benchmark
  - {name: "PTC_FM", validation_folds: 10, experiment_config_file: "Examples/TUExample/Configs/config_experiment.yml", type: "TUDataset"}
  # in case of a dataset in the correct format (the path to the data is given in the experiment config file)
  - {name: "CSL", validation_folds: 5, experiment_config_file: "Reproduce_RuleGNN/Configs/config_CSL.yml"}
```
### Experiment Config File

The experiment config file defines the hyperparameters, the model to use and all paths (to the data, proprocessing results, etc.).
For each dataset, you need to link an experiment config file in the main config file using the key ```experiment_config_file```.

## Data Format

The graph dataset is represented using three files:
1. `_Nodes.txt` containing the node features of each graph
    - each line represents a node in a graph of the dataset and is formatted as follows:
        ```
        graph_id node_id node_label (int, optional) node_feature_1 (float, optional) node_feature_2 (float, optional) ...
        ```
      where `graph_id` is the id of the graph the node belongs to, `node_id` is the id of the node in the graph, `node_label` is the integer label of the node, and `node_feature_i` are additional features of the node. If no node label is given all labels are set to 0.

2. `_Edges.txt` containing the edges of each graph
    - each line represents an edge in a graph of the dataset and is formatted as follows:
        ```
        graph_id node_id_1 node_id_2 edge_label (int, optional) edge_feature_1 (float, optional) edge_feature_2 (float, optional) ...
        ```
      where `graph_id` is the id of the graph the edge belongs to, `node_id_1` and `node_id_2` are the ids of the nodes the edge connects, `edge_label` is the integer label of the edge, and `edge_feature_i` are additional features of the edge. If no edge label is given all labels are set to 0.
3. `_Labels.txt` containing the labels of each graph
    - each line represents a graph in the dataset and is formatted as follows:
        ```
        graph_name graph_id graph_label (int or float)
        ```
      where `graph_name` is the name of the graph, `graph_id` is the id of the graph, and `graph_label` is the label of the graph.

## Layers
At the moment, the following layers are implemented:
- WL-Layer
  ```yaml
  - { layer_type: wl, wl_iterations: 2, max_node_labels: 500, properties: { name: distances, values: [1] }}
  ```
    The WL-Layer uses the Weisfeiler-Lehman algorithm to generate node labels. The parameter ```wl_iterations``` specifies the number of iterations of the Weisfeiler-Lehman algorithm. The parameter ```max_node_labels``` specifies the maximum number of node labels used in the layer. The parameter ```properties``` specifies the inter-node properties used, see [Property Functions](#Property-Functions).
- Subgraph-Layer
    ```yaml
    - { layer_type: subgraph, id: 0, properties: { name: distances, values: [ 3 ] }}
    ```
    For the subgraph layer, you need to specify under the keyword ```subgraph```  the list of subgraphs as nx.Graph objects, e.g.
   ```yaml
   subgraphs:
     - "[nx.cycle_graph(4), nx.cycle_graph(5)]"
    ```
  The parameter ```id``` specifies which list of subgraphs to use.
  In this example the layer uses the labels of the nodes induced by the embeddings of the subgraphs (in this case cycles of length 4 and 5).
  
- Cycle-Layer (special case of Subgraph-Layer)
  ```yaml
    - { layer_type: simple_cycles, max_cycle_length: 10, properties: { name: distances, values: [1,2,3,4,5,6] }}
    ```
  generates the node labels using the embeddings of simple_cycles of length 1 to 10.
  ```yaml
   - { layer_type: induced_cycles, max_cycle_length: 10, properties: { name: distances, values: [1,2,3,4,5,6] }}
  ```
    generates the node labels using the embeddings of induced_cycles of length 1 to 10.
- Cliques-Layer (special case of Subgraph-Layer)

## Property Functions

## Add new labeling functions

## Add new property functions

## Plotting