#!/bin/sh
DATA_PATH="/home/mlai21/seiffart/Data/GraphData/DS_all/"
RESULTS_PATH="/home/mlai21/seiffart/Results/RuleGNN/"
DISTANCES_PATH="/home/mlai21/seiffart/Data/GraphData/DS_all/Distances/"
DATABASE_NAME="DD"
MAX_RUN_ID=2
CROSS_VALIDATION=9
#NETWORK_TYPES="100;wl_1:1,2,3;wl_1 200;wl_1:1,2,3;wl_1 500;wl_1:1,2,3;wl_1 100;wl_2:1,2,3;wl_2 200;wl_2:1,2,3;wl_2 500;wl_2:1,2,3;wl_2 100;wl_1:1,2,3,4,5,6;wl_1 100;wl_2:1,2,3,4,5,6;wl_2 500;wl_2:1,2,3,4,5,6;wl_2 100;wl_1:1,2,3;wl_1:1,2,3;wl_1 500;wl_2:1,2,3;wl_2:1,2,3;wl_2"
#NETWORK_TYPES="100;wl_2:1,2,3;wl_2 200;wl_2:1,2,3;wl_2 500;wl_2:1,2,3;wl_2"
#DD NETWORK_TYPES="100;primary:1,2;100;primary"
#COLLAB NETWORK_TYPES="500;wl_0:1,2;500;wl_0"
#IMDB-BINARY NETWORK_TYPES="500;wl_0:1,2,3;500;wl_0"
#IMDB-MULTI NETWORK_TYPES="500;wl_0:1,2,3;500;wl_0"
#REDDIT-BINARY NETWORK_TYPES="500;wl_0:1,2,3;500;wl_0"
#REDDIT-MULTI-5K NETWORK_TYPES="500;wl_0:1,2,3;500;wl_0"
#PROTEINS NETWORK_TYPES="500;wl_1:1,2,3;wl_1"
#DHFR, Mutagenicity, NCI109 NETWORK_TYPES="500;wl_2:1,2,3,4,5,6;wl_2 500;wl_2:1,2,3;wl_2:1,2,3;wl_2 500;wl_2:1,2,3,4,5;wl_2"
#SYNTHETICnew NETWORK_TYPES="500;wl_1:1,2,3,4,5,6;wl_1 500;wl_1:1,2,3;wl_1:1,2,3;wl_1 500;wl_1:1,2,3,4,5;wl_1"
EPOCHS=50
BATCH_SIZE=16
EDGE_LABELS=-1
USE_FEATURES=True
USE_ATTRIBUTES=False
LEARNING_RATE=0.001
BALANCED=False
LOAD_SPLITS=True
CONVOLUTION_GRAD=True
RESIZE_GRAD=True

for network_type in $NETWORK_TYPES; do
  (for i in $(seq 0 $MAX_RUN_ID); do
      (for j in $(seq 0 $CROSS_VALIDATION); do
        python GraphLearningMain.py --data_path $DATA_PATH --results_path $RESULTS_PATH --distances_path $DISTANCES_PATH --graph_db_name $DATABASE_NAME --run_id $i --validation_id $j --network_type $network_type --epochs $EPOCHS --batch_size $BATCH_SIZE --edge_labels $EDGE_LABELS --use_features $USE_FEATURES --use_attributes $USE_ATTRIBUTES --lr $LEARNING_RATE --load_splits $LOAD_SPLITS  --balanced $BALANCED --convolution_grad $CONVOLUTION_GRAD --resize_grad $RESIZE_GRAD --mode fast &
      done
      wait)
  done
  wait)
done

