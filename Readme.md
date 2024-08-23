# Using RuleGNNs for Graph Classification

This repository contains an explanation on how to reproduce the results of the below paper.
Moreover, we explain how to use RuleGNNs for custom datasets and new rules.

# [Rule Based Learning with Dynamic (Graph) Neural Networks](https://arxiv.org/abs/2406.09954)

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

All important information about the graph dataset is separated into 
 - the raw data ```graph structure``` including the node features (see [Data Format](##Data Format)) and the ```labels``` for each graph.
 - different ```labels``` for the nodes of each graph
 - different ```properties``` between pairs of nodes in the graph
## Data Format

The graph dataset is represented using three files:
- `_Nodes.txt` containing the node features of each graph

### Multi-Class Classification vs Regression

## Rules

## Run the experiments

## Evaluation

## Plotting