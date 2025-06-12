#!/bin/bash

# set PATH
export PYTHONPATH=/home/mlai21/share/code/RuleGNN/
# set number of threads variable
NUM_THREADS=30

# if NUM_THREADS > 1, set OMP_NUM_THREADS to 1
if [ $NUM_THREADS -gt 1 ]; then
    export OMP_NUM_THREADS=1
fi


export OMP_NUM_THREADS=1
# activate conda
eval "$(conda shell.bash hook)"
conda activate RuleGNN

# run the script
python Reproduce/experiments_all.py --num_threads $NUM_THREADS