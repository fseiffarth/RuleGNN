# Benchmarking GNNs

This repository should be a starting point to compare various GNN architectures on various dataset.
It is based on the pytorch geometric representation of graphs and implements various evaluation schemes including the the fair evaluation of
GNNs as proposed in the paper [Fair Setup for Benchmarking Graph Neural Networks](https://arxiv.org/abs/2309.14924) by Errica et al.

The implemented framework mainly bases on two configuration files:
1. In the [main config file](Examples/ConfigurationFiles/example_config_main.yml) you can define which datasets to use and how many splits which splits are used for training, validation, and test sets.
2. In the [network config file](Examples/ConfigurationFiles/example_config_experiment.yml) you can define the architecture to use and the corresponding hyperparameters of the network.

The links above lead to some example configuration files that can be used as a starting point for your own experiments.
If running experiments with the framework, the results will be automatically saved in a clearly structured way.
Below we provide a detailed description of how to set up the environment, reproduce the experiments of our paper, and customize your own experiments with your favorite datasets and architectures.

### List of implemented GNN architectures
- Graph Convolutional Network (GCN) 
- Graph Attention Network (GAT)
- Graph Isomorphism Network (GIN)
- GraphSAGE
- ShareGNN

### Table of implemented datasets
The following datasets are automatically downloaded and processed.
Custom datasets can be added as described in [Customize Experiments](#Customize-Experiments).

#### Graph Classification 
Use ```task: graph_classification``` in config file

  **Real World Graphs**

  | name           | type | Source | Comments |
  |----------------|----------------------------------|--------------------------------|------------|
  | All TUDatasets | TUDataset                        | https://chrsmrrs.github.io/datasets/docs/datasets/ | |
  | e.g            |                                  || |
  | MUTAG          | TUDataset                        | https://chrsmrrs.github.io/datasets/docs/datasets/ | |
  | NCI1           | TUDataset                        | https://chrsmrrs.github.io/datasets/docs/datasets/ | |
  | DHFR           | TUDataset                        | https://chrsmrrs.github.io/datasets/docs/datasets/ | |

  **Synthetic Graphs**

  | name | type | Source | Comments |
  |----------------|----------------------------------|--------------------------------|------------|
  | CSL | gnn_benchmark | --- | |
  | Snowflakes | generate_from_function | --- | |
  | EvenOddRings2_16 | generate_from_function |  | |
  | EvenOddRingsCount16 | generate_from_function |  | |
  | LongRings100 | generate_from_function |  | |


#### Graph Regression
Use ```task: graph_regression``` in config file

| name | type  | Source | Comments |
| ------- |--------------------------------| ----------------------------| ------------|
| ZINC-12k | ZINC                        | | |
| ZINC-250k | ZINC                      | | |
| Substructure Counting Benchmark | SubstructureBenchmark |  | |


#### Node Classification 
Use ```task: node_classification``` in config file

name | type | Source | Comments |
| ------- |--------------------------------| -------------------------------| ------------|
| Cora | Planetoid | https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#planetoid | |
| Citeseer | Planetoid | https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#planetoid | |
| Pubmed | Planetoid | https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#planetoid | |
| Nell | Nell | https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#nell | |
| ogbn-arxiv | ogbn | https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv | |

- **Link Prediction:**
  (not implemented yet)



## Setting up the Environment

1. Clone the repository

2. Install the required packages using the install.sh script:
   ```bash
   # Make the script executable (this is necessary before you can run it)
   chmod +x install.sh
   # Run the installation script
   ./install.sh
   ```

   > **Note:** The `chmod +x install.sh` command makes the script executable. This is a necessary step on Unix-based systems (Linux/macOS) before you can run a shell script. If you're on Windows using Git Bash or WSL, you'll also need this command.

   This script will:
   - Check if Python 3.12 is installed (with installation hints if it's not)
   - Create a Python virtual environment
   - Install all dependencies from requirements.txt
3. **(for command line)** To run the scripts with the correct paths please set your PYTHONPATH (working directory) to the root directory of the repository.
   ```bash
   export PYTHONPATH=/path/to/ShareGNN
   ```
    **(for IDE)** If you are working in an IDE, you can set the PYTHONPATH in the run configuration. 
   E.g., in PyCharm, you have to change the working directory path to the root directory of the repository.
    Go to ```File -> Settings -> Project Structure``` and mark the root directory as ```Sources``` (blue folder icon).


## Reproduce ShareGNN Paper Experiments
To reproduce the experiments of the paper, follow the steps below. All necessary code can be found in the [Reproduce](Reproduce) folder.
All experiments take approximately 4 days on an AMD Ryzen 9 7950X with 16 cores and 32 threads and 128 GB of RAM.
Also single experiments can be started, see [Run specific experiment](#Run-specific-experiment).

### Run all experiments
```bash
python Reproduce/experiments_all.py --num_threads 30
```

The following steps are executed:

   - download of the datasets
   - preprocessing of the datasets
   - experiments regarding fair evaluation, the standard evaluation, the synthetic data, the baselines and the ablation experiments
   - grid search to find the best hyperparameters for different models
   - best models three times with different seeds
   - evaluation of the results
   - creation of the Feature Data for the best runs


All results will be saved in the [Reproduce/Results](Reproduce/Results) folder.

For each experiment and each dataset the following evaluation files are produced:

- ```summary.csv```: contains the results of the grid search (fair evaluation) one row per hyperparameter setting
- ```summary_best.csv```: contains the results of the best model (hyperparameter setting) one row per seed
- ```summary_best_mean.csv```: contains the mean and standard deviation of the best model results over all seeds

At the end of the experiments there will exist the folder [Reproduce/DataGNNComparison](Reproduce/DataGNNComparison) containing the
graph data with the labels from the best ShareGNN run.
These graph data can be directly used further for the competitors, see https://anonymous.4open.science/r/FairSetup-F3DE/CONTRIBUTING.md for the details

### Visualize the results
The results will be saved under ```Reproduce/Results/Latex/Plots/```.
```bash
python Reproduce/latex_plots.py
```

### Get Latex Tables
Note, that to get the full tables of the paper also the competitors need to be run, see https://anonymous.4open.science/r/FairSetup-F3DE/CONTRIBUTING.md.
Then copy the RESULTS folder to this repository.
```bash
python Reproduce/latex.py
```










### Run specific experiment
1. Real World Fair Evaluation
    ```bash
    python Reproduce/experiments_fair_real_world.py --num_threads 30
    ```
2. Real World Standard Evaluation
    ```bash
    python Reproduce/experiments_standard_real_world.py --num_threads 30
    ```
3. Synthetic Fair Evaluation
    ```bash
    python Reproduce/experiments_synthetic.py --num_threads 30
    ```
4. Baseline Comparison
    ```bash
    python Reproduce/experiments_baseline.py --num_threads 30
    ```

5. Distance/Layer Ablation
    ```bash
    python Reproduce/experiments_distance_ablation.py --num_threads 30
    ```

6. Number of Weights Ablation
    ```bash
    python Reproduce/experiments_threshold_ablation.py --num_threads 30
    ```



## Experiments on the TU Dortmund Graph Benchmark
All datasets from the TU Dortmund Benchmark available [here](https://chrsmrrs.github.io/datasets/docs/datasets/) can be used directly for experiments as shown in [Examples/TUExample](Examples/TUExample).

To run the example use:
```bash
python Examples/TUExample/main.py
```


## Customize Experiments
An example of how to use ShareGNNs for custom datasets can be found in [Examples/CustomExample](Examples/CustomExample).
Most importantly, your dataset needs to be in the correct format.
At the moment, the code supports three different options.

- Option 1: Use a pytorch geometric dataset

- Option 2: Save your favorite graph dataset in the format described below in [Data Format](#Data-Format).

- Option 3: Add your function ```favorite_graph_dataset_generator``` to [src/utils/SyntheticGraphs.py](src/utils/SyntheticGraphs.py) 
   that returns a tuple of the form 
    ```
      (List[networkx.Graph], List[int/float])
   ```
   where the first list contains the networkx graphs (optional with node and edge labels) and the second list contains the labels of the graphs. 

All the experiment details are defined in two configuration files:
- the [main config file](Examples/ConfigurationFiles/example_config_main.yml)  that defines which datasets you want to use and how many splits are used for validation
- the [experiment config file](Examples/ConfigurationFiles/example_config_experiment.yml) that defines the hyperparameters, the model to use and all paths (to the data, Preprocessing results, etc.)

To run an experiment you only need the following code:

   ```python
   from pathlib import Path

from src.Experiment.ExperimentMain import ExperimentMain


def main():
    experiment = ExperimentMain(Path('Path/To/Your/Main/Config/File.yml'))
    experiment.ExperimentPreprocessing()
    experiment.GridSearch()
    experiment.EvaluateResults()
    experiment.RunBestModel()
    experiment.EvaluateResults(evaluate_best_model=True)


if __name__ == '__main__':
    main()
   ```
- In the preprocessing step ```experiment.Preprocess()```, the data will be downloaded or generated and labels and properties according to the experiment configuration file described below are precomputed.
- Then in ```experiment.GridSearch()```, the best model hyperparameters for the dataset are found using a 10-fold cross-validation.
- Finally, in ```experiment.RunBestModel()```, the best model is trained three times with different seeds and evaluated on the test set.
- The results are evaluated in ```experiment.EvaluateResults()``` saved as ```summary.csv``` resp. ```summary_best.csv``` and ```summary_best_mean.csv``` in the results folder specified in the experiment configuration file under the database name.


### Main Config File
The main config file defines which datasets to use and how to split the data into training, validation, and test sets.
Moreover, all hyperparameters of the network are defined here.

### Experiment Config File

The experiment config defines the hyperparameters for the layers of the model



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
At the moment, the following layers are implemented.
The list of parameters always iterates over all combinations.
- Primary-Layer
  ```yaml
  - { label_type: primary }
  ```
  The primary layer uses the initial node labels.
- WL-Layer
  ```yaml
  - { label_type: wl, depth: [ 0,1,2,3,4 ] },
  ```
    The WL-Layer uses the Weisfeiler-Lehman algorithm to generate node labels. 
    The parameter ```depth``` specifies the number of iterations of the Weisfeiler-Lehman algorithm. 
    The parameter ```max_labels``` specifies the maximum number of node labels used in the layer. If not given the number is unlimited.
- Subgraph-Layer
    ```yaml
  - { label_type: subgraph, id: [ 0,1,2,3 ] }
    ```
    For the subgraph layer, you need to specify under the keyword ```subgraph```  the list of subgraphs as nx.Graph objects, e.g.
   ```yaml
    subgraphs:
    - "[nx.complete_graph(4)]"
    - "[nx.cycle_graph(3), nx.star_graph(1)]"
    - "[nx.cycle_graph(4), nx.star_graph(1)]"
    - "[nx.cycle_graph(3), nx.cycle_graph(4), nx.star_graph(1)]"
  ```
  The parameter ```id``` specifies which list of subgraphs to use.
  In this example the layer uses the labels of the nodes induced by the embeddings of the subgraphs (in this case cycles of length 4 and 5).

- Cycle-Layer (special case of Subgraph-Layer)
  ```yaml
    - { label_type: simple_cycles, max_cycle_length: [ 3,4,5 ] },
    ```
  generates the node labels using the embeddings of simple_cycles of length 1 to 10.
  ```yaml
   - { label_type: induced_cycles, max_cycle_length: [ 4,5,10,20 ] },
  ```
    generates the node labels using the embeddings of induced_cycles of length 1 to 10.
- Cliques-Layer (special case of Subgraph-Layer)
    ```yaml
        - { label_type: cliques, max_clique_size: [ 3,4,6,10,20,50 ] },
    ```
    generates the node labels using the embeddings of cliques of size 1 to 10.

## Property Functions

The property functions assign each pair of nodes in a graph a property value. This can be distances, information about edge labels between the nodes or different values if one node is in a circle and the other is not.
In the paper the property functions are always distances.
At the moment, the following property functions are implemented:
- Distances
    ```yaml
    - properties: [
          { name: distances, values: [ 0,1,2 ] },
        ]
    ```
    Each distance in values is considered. If the distance between two nodes is not in the values list, then no corresponding learnable parameter (weight) is created.
- Edge Label Distances
    ```yaml
    - properties: [
          { name: edge_label_distances, values: [ 0,1,2 ] },
        ]
    ```
    Not only distances are considered but also the counts of all edge labels between all shortest paths of the nodes.
- Circle Distances
    ```yaml
    - properties: [
          { name: circle_distances, values: [ 0,1,2,3 ] },
        ]
    ```
    For each distance depicted in values, there are four different values: 0 if both nodes are not in a circle, 1 if both nodes are in a circle, 2 if only the first node is in a circle, and 3 if only the second node is in a circle.

## Add new labeling functions
To define a new node labeling function, go to [src/Preprocessing/create_labels.py](src/Preprocessing/create_labels.py) and add a new function called ```save_<your_labeling_function>_labels```.
The node labels should be generated as list of lists of integers (one list of node labels per graph). 
Moreover, give your new labeling function a unique ```label_type``` used as argument in the config file.
- **Save Labels:** Use ```save_labels_to_file(file, graph_data.name, l, graph_node_labels, max_labels)``` to save the labels to the path ```file```. 
The filename will be ```graph_data.name_l.pt```.
- **Load Labels**: Go to [src/Architectures/ShareGNN/ShareGNNLayers.py](src/Architectures/ShareGNN/ShareGNNLayers.py) and add a new case to the function ```get_labels_string``` that gives you the string ```<your_characteristic_labeling_function_string>``` for your labeling function based on possible additional arguments.
- **Automatic Label Generation**: If you want to automatically generate the labels based on the config file you need to go to
[src/Preprocessing/DatasetPreprocessing.py](src/Preprocessing/DatasetPreprocessing.py) and add a new case in the function ```layer_to_labels``` that calls your labeling function based on the ```label_type``` given in the config file.

## Add new property functions
To define a new property function, go to [src/Preprocessing/create_properties.py](src/Preprocessing/create_properties.py) and add a new function called ```write_<your_property_function>_properties```.
Moreover, give your new property function a unique ```properties``` key used as argument in the config file.
- **Save Properties:**
- **Load Properties:**
- **Automatic Property Generation**: If you want to automatically generate the properties based on the config file you need to go to 
[scripts/Preprocessing.py](src/Preprocessing/Preprocessing.py) and add a new case in the function ```property_to_properties``` that calls your property function based on the ```properties``` given in the config file.

## Plotting

For plotting you need ```graphviz```
See [Reproduce/latex_plots.py](Reproduce/latex_plots.py) for an example of how to plot the graphs.
