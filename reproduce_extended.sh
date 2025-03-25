#!/bin/bash

# set PATH
export PYTHONPATH=/home/mlai21/share/code/RuleGNN/
# activate conda
eval "$(conda shell.bash hook)"
conda activate RuleGNN

# run the script
python ReproduceExtended/main.py --num_threads 30