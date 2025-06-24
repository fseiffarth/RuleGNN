#!/bin/bash

# set PATH to the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH=$SCRIPT_DIR
# set number of threads variable
NUM_THREADS=30

# if NUM_THREADS > 1, set OMP_NUM_THREADS to 1
if [ $NUM_THREADS -gt 1 ]; then
    export OMP_NUM_THREADS=1
fi


export OMP_NUM_THREADS=1
# check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

# activate virtual environment
source venv/bin/activate

# run the script
python Reproduce/experiments_all.py --num_threads $NUM_THREADS
