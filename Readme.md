# RuleGNN

This repository contains the code for experiments with RuleGNNs as described in the paper [Rule Based Learning with Dynamic (Graph) Neural Networks](https://arxiv.org/abs/2406.09954).
First, we give an overview of the repository and how to reproduce the experiments of the paper.
Then, we explain how to use RuleGNNs for [custom datasets](#Customize-Experiments) and how to add new [layers](#Layers), [labeling functions](#Add-new-labeling-functions), and [property functions](#Add-new-property-functions).

## Setting up the Environment

1. Clone the repository using
    ```bash
    git clone git@github.com:fseiffarth/RuleGNN.git
    ```
2. Install the required packages using the environment.yml file using the following command:
   ```bash
   conda env create -f environment.yml
   ```
3. **(for command line)** To run the scripts with the correct paths please set your PYTHONPATH (working directory) to the root directory of the repository.
   ```bash
   export PYTHONPATH=/path/to/RuleGNN
   ```
    **(for IDE)** If you are working in an IDE, you can set the PYTHONPATH in the run configuration. E.g., in PyCharm, you have to change the working directory path to the root directory of the repository.

## Reproduce RuleGNN Experiments
To reproduce the experiments of the paper, follow the steps below. All necessary code can be found in the [Reproduce_RuleGNN](Reproduce_RuleGNN) folder.
The experiments take approximately 2 days on an AMD Ryzen 9 7950X with 16 cores and 32 threads and 128 GB of RAM.

The commands
```bash
python Reproduce_RuleGNN/main_fair.py
```

```bash
python Reproduce_RuleGNN/main_sota.py
```
```bash
python Reproduce_RuleGNN/main_sota_random.py
```

will run the experiments in the fair evaluation and the state-of-the-art evaluation, respectively.

The following steps are executed:

   - download the datasets
   - preprocess the datasets
   - run the grid search to find the best hyperparameters for different models
   - run the best models three times with different seeds
   - evaluate the results

All results related to the fair evaluation will be saved in the [Reproduce_RuleGNN/Results](Reproduce_RuleGNN/Results) folder.
All results related to the state-of-the-art evaluation will be saved in the [Reproduce_RuleGNN/Results_SOTA](Reproduce_RuleGNN/ResultsSOTA) folder.

The following evaluation files are produced:
    - ```summary.csv```: contains the results of the grid search (fair evaluation) one row per hyperparameter setting
    - ```summary_best.csv```: contains the results of the best model (hyperparameter setting) one row per seed
    - ```summary_best_mean.csv```: contains the mean and standard deviation of the best model results over all seeds

To visualize the results run:
```bash
python Reproduce_RuleGNN/plotting.py
```
The results will be saved in the corresponding ```Plots``` folder under ```Reproduce_RuleGNN/Results/<DB_NAME>```.

## Experiments on the TU Dortmund Graph Benchmark
All datasets from the TU Dortmund Benchmark available [here](https://chrsmrrs.github.io/datasets/docs/datasets/) can be used directly for experiments as shown in [Examples/TUExample](Examples/TUExample).

## Customize Experiments
An example of how to use RuleGNNs for custom datasets can be found in [Examples/CustomExample](Examples/CustomExample).
Most importantly, your dataset needs to be in the correct format.
At the moment, the code supports two different options.

- Option 1 (preferred): Save your favorite graph dataset in the format described below in [Data Format](#Data-Format).

- Option 2: Add your function ```favorite_graph_dataset_generator``` to [src/utils/SyntheticGraphs.py](src/utils/SyntheticGraphs.py) 
   that returns a tuple of the form 
    ```
      (List[networkx.Graph], List[int/float])
   ```
   where the first list contains the networkx graphs (optional with node and edge labels) and the second list contains the labels of the graphs. 

All the experiment details are defined in two configuration files:
- the [main config file](#main-config-file)  that defines which datasets you want to use and how many splits are used for validation
- the [experiment config file](#experiment-config-file) that defines the hyperparameters, the model to use and all paths (to the data, proprocessing results, etc.)

To run the experiment you only need the following code:

   ```python
   from pathlib import Path

from scripts.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Path/To/Your/Main/Config/File.yml'))
    experiment.Preprocess()
    experiment.GridSearch()
    experiment.RunBestModel()
    experiment.EvaluateResults()
    experiment.EvaluateResults(evaluate_best_model=True)


if __name__ == '__main__':
    main()
   ```
- In the preprocessing step ```experiment.Preprocess()```, the data will be downloaded or generated and labels and properties according to the experiment configuration file described below are precomputed.
- Then in ```experiment.GridSearch()```, the best model hyperparameters for the dataset are found using a 10-fold cross-validation.
- Finally, in ```experiment.RunBestModel()```, the best model is trained three times with different seeds and evaluated on the test set.
- The results are evaluated in ```experiment.EvaluateResults()``` saved as ```summary.csv``` resp. ```summary_best.csv``` and ```summary_best_mean.csv``` in the results folder specified in the experiment configuration file under the database name.


### Main Config File
In the main config file, you define which datasets you want to use and how to split the data into training, validation, and test sets.
The file should look like this:
```yaml
datasets:
  # in case of a given generation function called ring_diagonals in this case
  - {name: "EXAMPLE_DB", data: "Examples/CustomExample/Data/SyntheticDatasets/", validation_folds: 10, experiment_config_file: "Examples/CustomExample/Configs/config_experiment.yml", type: "generate_from_function", generate_function: ring_diagonals, generate_function_args: {data_size: 1000, ring_size: 50}}
  # in case of a dataset from the TU Dortmund Benchmark
  - {name: "PTC_FM", data: "Reproduce_RuleGNN/Data/TUDatasets/",, validation_folds: 10, experiment_config_file: "Examples/TUExample/Configs/config_experiment.yml", type: "TUDataset"}
  # in case of a dataset in the correct format (the path to the data is given in the experiment config file)
  - {name: "CSL",data: "Reproduce_RuleGNN/Data/SyntheticDatasets/", validation_folds: 5, experiment_config_file: "Reproduce_RuleGNN/Configs/config_CSL.yml"}

paths:
  # all the paths are relative to the PYTHONPATH path, can be also defined dataset-wise in the experiment_config_file
  properties:
    "Reproduce_RuleGNN/Data/Properties/" # Precomputed properties will be loaded from this folder
  labels:
    "Reproduce_RuleGNN/Data/Labels/" # Path to the folder containing the labels
  splits:
    "Reproduce_RuleGNN/Data/Splits/" # Path to the folder containing the data splits
  results:
    "Reproduce_RuleGNN/Results/" # Results will be saved in this folder

```
The following keys are used:
- ```name```: the name of the dataset
- ```data```: the path to the folder containing the graph data or where the data will be saved if generated or downloaded
- ```validation_folds```: the number of splits used for validation
- ```experiment_config_file```: the path to the experiment config file
- ```type``` (optional): the type of the dataset, if not given the dataset is assumed to be in the correct format in the path given in the experiment config file, if given it should be one of the following:
  - ```generate_from_function```: the dataset is generated using a function defined in [src/utils/SyntheticGraphs.py](src/utils/SyntheticGraphs.py)
  - ```TUDataset```: the dataset is from the TU Dortmund Benchmark
- ```generate_function``` (optional): the name of the function used to generate the dataset if the type is ```generate_from_function```
- ```generate_function_args``` (optional): the arguments of the function used to generate the dataset as a dictionary if the type is ```generate_from_function```

The paths key defines where to save the precomputed properties, labels, splits, and results. 
If not given, the paths are assumed to be in the path given in the experiment config file.

### Experiment Config File

The experiment config file defines the hyperparameters, the model to use and all paths (to the data, proprocessing results, etc.).
For each dataset, you need to link an experiment config file in the main config file using the key ```experiment_config_file```.
```yaml
paths:
  data:
    "Examples/CustomExample/Data/" # Path to the folder containing the graph data
  properties:
    "Examples/CustomExample/Data/Properties/" # Precomputed properties will be loaded from this folder
  labels:
    "Examples/CustomExample/Data/Labels/" # Path to the folder containing the labels
  results:
    "Examples/CustomExample/Results/" # Results will be saved in this folder
  splits:
    "Examples/CustomExample/Data/Splits/" # Path to the folder containing the data splits

device: # cpu or cuda, cpu is recommended for the experiments mode as it is faster at the moment
  cpu
mode:
  experiments # if debug printing and plotting options are enabled, for the experiments mode should be 'experiments'
batch_size:
  - 128
learning_rate:
  - 0.05
epochs:
  - 10
scheduler:
  False
dropout:
  - 0.0
optimizer:
  - Adam
loss:
  - CrossEntropyLoss
early_stopping:
  enabled:
    False
  patience:
    25
networks:
  #- - { layer_type: primary, properties: { name: edge_label_distances, values: [ 1 ] } }
  #  - { layer_type: wl, wl_iterations: 0, properties: { name: distances, values: [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 ] } }
  #  - { layer_type: wl, wl_iterations: 0 }

  # wl model
  - - { layer_type: wl, wl_iterations: 2, max_node_labels: 500, properties: {name: distances, values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]} }
    - { layer_type: wl, wl_iterations: 2, max_node_labels: 500 }



use_features: # if True uses normlized node labels as input features, if False uses 1-vector as input features
  False
use_attributes: # if True uses node attributes instead of node labels
  False
random_variation: # if True adds random variation to the input features
  False
load_splits: # if True loads precomputed data splits (use False only for new datasets)
  True

# data options
balance_training:
  False

# Additional options for analysis only possible in debug mode
additional_options:
  draw: # draw the accuracy and loss during training
    True
  save_weights: # save the weights of the model
    False
  save_prediction_values:
    False
  plot_graphs: # Plot all graphs in the dataset
    False
  print_results: # Print accuracy and loss in the console
    True


prune:
  enabled:
    False
  epochs: # prune after this many epochs
    25
  percentage: # number of total weights pruned at the end of training per layer (0.1 means 10% of the weights will be pruned)
    - 0.999
    - 0.5

precision:
  double

best_model:
  False
save_last:
  False
```
The current available keys are:
- ```paths```: [optional] will overwrite the paths in the main config, the paths to the data, properties, labels, results, and splits
  - ```data```: the path to the folder containing the graph data or where the data will be saved if generated or downloaded
  - ```properties```: the path to the folder containing the precomputed properties
  - ```labels```: the path to the folder containing the labels
  - ```splits```: the path to the folder containing the data splits  
  - ```results```: the path to the folder where the results will be saved
- ```device```: the device used for training, either 'cpu' or 'cuda' (recommended: 'cpu')
- ```mode```: the mode of the experiment, either 'experiments' or 'debug' (recommended: 'experiments' for experiments and 'debug' for debugging)
- ```batch_size```: the batch size used for training
- ```learning_rate```: the learning rate used for training
- ```epochs```: the number of epochs used for training
- ```scheduler```: if True, a scheduler is used
- ```dropout```: the dropout rate used for training (not tested yet)
- ```optimizer```: the optimizer used for training
- ```loss```: the loss function used for training
- ```early_stopping```: if True, early stopping is used
 - ```enabled```: if True, early stopping is enabled
 - ```patience```: the patience of the early stopping, after how many epochs without improvement of the validation accuracy the training stops
- ```networks```: the network architecture used for training (see [Layers](#Layers)) for more details)
- ```use_features```: if True, uses the normalized node labels as input features, if False uses the 1-vector as input features
- ```use_attributes```: if True, uses node attributes instead of node labels
- ```random_variation```: if True, adds random variation to the input features
- ```balance_training```: if True, balances the training set
- ```additional_options```: additional options for analysis only possible in debug mode
 - ```draw```: if True, draws the accuracy and loss during training
 - ```save_weights```: if True, saves the weights of the model
 - ```save_prediction_values```: if True, saves the prediction values
 - ```plot_graphs```: if True, plots all graphs in the dataset
 - ```print_results```: if True, prints accuracy and loss in the console
 - ```prune```: if True, prunes the model
   - ```enabled```: if True, pruning is enabled
   - ```epochs```: prune after this many epochs
   - ```percentage```: the number of total weights pruned at the end of training per layer (0.1 means 10% of the weights will be pruned)
- ```precision```: the precision used for training
- ```best_model```: if True, also test accuracy is evaluated for all models
- ```save_last```: if True, saves the last model



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
- Primary-Layer
  ```yaml
  - { layer_type: primary, properties: { name: edge_label_distances, values: [ 1 ] } }
  ```
  The primary layer uses the initial node labels.
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
   - { layer_type: induced_cycles, max_cycle_length: 10, max_node_labels: 500, properties: { name: distances, values: [1,2,3,4,5,6] }}
  ```
    generates the node labels using the embeddings of induced_cycles of length 1 to 10.
- Cliques-Layer (special case of Subgraph-Layer)
    ```yaml
        - { layer_type: cliques, max_clique_size: 10, max_node_labels: 500, properties: { name: distances, values: [1,2,3,4,5,6] }}
    ```
    generates the node labels using the embeddings of cliques of size 1 to 10.

## Property Functions

The property functions assign each pair of nodes in a graph a property value. This can be distances, information about edge labels between the nodes or different values if one node is in a circle and the other is not.
At the moment, the following property functions are implemented:
- Distances
    ```yaml
    - properties: { name: distances, values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25] }
    ```
    Each distance in values is considered. If the distance between two nodes is not in the values list, then no corresponding learnable parameter (weight) is created.
- Edge Label Distances
    ```yaml
    - properties: { name: edge_label_distances, values: [1] }
    ```
    Not only distances are considered but also the counts of all edge labels between all shortest paths of the nodes.
- Circle Distances
    ```yaml
    - properties: { name: circle_distances, values: [1] }
    ```
    For each distance depicted in values, there are four different values: 0 if both nodes are not in a circle, 1 if both nodes are in a circle, 2 if only the first node is in a circle, and 3 if only the second node is in a circle.

## Add new labeling functions
To define a new node labeling function, go to [src/Preprocessing/create_labels.py](src/Preprocessing/create_labels.py) and add a new function called ```save_<your_labeling_function>_labels```.
The node labels should be generated as list of lists of integers (one list of node labels per graph). 
Moreover, give your new labeling function a unique ```label_type``` used as argument in the config file.
- **Save Labels:** Use ```write_node_labels(file, node_labels)``` to save the labels to the path ```file```.
The file name should be ```<DB_NAME>_<your_characteristic_labeling_function_string>_labels.txt```.
- **Load Labels**: Go to [src/Architectures/RuleGNN/RuleGNNLayers.py](src/Architectures/RuleGNN/RuleGNNLayers.py) and add a new case to the function ```get_layer_string``` that gives you the string ```<your_characteristic_labeling_function_string>``` for your labeling function based on possible additional arguments.
- **Automatic Label Generation**: If you want to automatically generate the labels based on the config file you need to go to
[scripts/Preprocessing.py](scripts/Preprocessing.py) and add a new case in the function ```layer_to_labels``` that calls your labeling function based on the ```label_type``` given in the config file.

## Add new property functions
To define a new property function, go to [src/Preprocessing/create_properties.py](src/Preprocessing/create_properties.py) and add a new function called ```write_<your_property_function>_properties```.
Moreover, give your new property function a unique ```properties``` key used as argument in the config file.
- **Save Properties:**
- **Load Properties:**
- **Automatic Property Generation**: If you want to automatically generate the properties based on the config file you need to go to 
[scripts/Preprocessing.py](scripts/Preprocessing.py) and add a new case in the function ```property_to_properties``` that calls your property function based on the ```properties``` given in the config file.

## Plotting

For plotting you need ```graphviz```

Use the following code to plot the learned parameters of the model:
```python
    db_name = 'DB_NAME'
    main_config_file = 'path/from/root/to/main_config_file.yml'
    experiment_config_file = 'path/from/root/to/experiment_config_file.yml'
    colormap = matplotlib.cm.get_cmap('viridis') # colormap for the weights
    graph_ids = [0,1,2] # list of graph ids to plot (test set ids
    filter_sizes = (None, 10, 3) #filter by largest absolute values, None means no filter
    # parameters for the graph drawing (node_size, edge_width, weight_edge_width, weight_arrow_size, colormap, etc.)
    # The first entry is for the original graph, the second for the learned parameters
    graph_drawing = (
        GraphDrawing(node_size=40, edge_width=1),
        GraphDrawing(node_size=40, edge_width=1, weight_edge_width=2.5, weight_arrow_size=10, colormap=colormap)
    ) 
    ww = WeightVisualization(db_name=db_name, main_config=main_config_file, experiment_config=experiment_config_file)
    for validation_id in range(10):
        for run in range(3):
            ww.visualize(graph_ids, run=run, validation_id=validation_id, graph_drawing=graph_drawing, filter_sizes=filter_sizes)
```