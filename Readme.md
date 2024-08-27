# Using RuleGNNs for Graph Classification

This repository contains an explanation on how to [reproduce](#Reproduce-RuleGNN-Experiments) the results of our paper [Rule Based Learning with Dynamic (Graph) Neural Networks](https://arxiv.org/abs/2406.09954).
Moreover, we give a detailed explanation on how to use RuleGNNs for [custom datasets](#Custom-Datasets).

# Reproduce RuleGNN Experiments

## Preparation:
1. Clone the repository
2. Install the required packages using the environment.yml file
3. Download the TUDataset from here https://chrsmrrs.github.io/datasets/docs/datasets/ and place the extracted folder in GraphData/TUGraphs/
4. Preprocess the distances by running GraphData/Distances/save_distances.py

## Run the experiments:
Using the ModelSelection.sh script, you can run the experiments for the different datasets and models. 
The DATABASE_NAMES and the CONFIG_FILE variable in the script can be adjusted to run the experiments for different datasets and models.

Possible combinations of DATABASE_NAMES and CONFIG_FILE are:

DATABASE_NAMES = ["NCI1", "NCI109", "Mutagenicity"] <br>
CONFIG_FILE = "Configs/config_NCI1.yml"

DATABASE_NAMES = ["DHFR"] <br>
CONFIG_FILE = "Configs/config_DHFR.yml"

DATABASE_NAMES = ["IMDB-BINARY", "IMDB-MULTI"] <br>
CONFIG_FILE = "Configs/config_IMDB-BINARY.yml"

DATABASE_NAMES = ["LongRings100"] <br>
CONFIG_FILE = "Configs/config_LongRings.yml"

DATABASE_NAMES = ["EvenOddRings2_16] <br>
CONFIG_FILE = "Configs/config_EvenOddRings.yml"

DATABASE_NAMES = ["EvenOddRingsCount16"] <br>
CONFIG_FILE = "Configs/config_EvenOddRingsCount.yml"

DATABASE_NAMES = ["Snowflakes"] <br>
CONFIG_FILE = "Configs/config_Snowflakes.yml"

Using the RunBestModels.sh script, the model with the best validation accuracy is chosen and the model is trained three times with different seeds.

## Evaluation
The results of the experiments can be evaluated running EvaluationFinal.py

# Custom Datasets

The easiest way to run RuleGNNs on custom datasets is if the dataset is in the [format](#Data-Format)
described below.
For an example of the format, see the files in [Example/Data/EXAMPLE_DB/raw](Examples/CustomDataset/Data/EXAMPLE_DB/raw).
The preprocessing steps are explained below.

## Preprocessing

Custom datasets can be created using the following four steps below.
We have created an example that can be found in the [Example](Examples) folder.
In the [Example/main.py](Examples/CustomDataset/main.py) file, we give the code for a detailed example of how to preprocess a custom dataset.
If your graph dataset is already in the correct format, you can skip the first step.
If your graph dataset is given in a different format use the [utils.py](utils.py) file to convert the dataset to the correct format.

1. Create the graph dataset (the data format is explained in the [Data Format](##Data Format) section)
2. Create the training, validation and test splits (use the function ```generate_splits``` in the [Example/main.py](Examples/CustomDataset/main.py) file)
3. Create the node labels (use the function ```generate_node_labels``` in the [Example/main.py](Examples/CustomDataset/main.py) file)
4. Create the properties between pairs of nodes in the graphs (use the function ```generate_properties``` in the [Example/main.py](Examples/CustomDataset/main.py) file)

## Run the experiments (using bash script only)
If the custom dataset is created as explained above, the experiments can be run using the ModelSelection.sh script.
1. Set the following variables in the ModelSelection.sh script:
   - DATABASE_NAMES = "Example"
   - CONFIG_FILE = "Example/Config/config_example.yml"
2. Run the ModelSelection.sh script by running the following command in the terminal:
   ```bash
   chmod u+x ModelSelection.sh
    ./ModelSelection.sh
   ```



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

### Multi-Class Classification vs Regression

## Rules

## Run the experiments

## Evaluation

## Plotting