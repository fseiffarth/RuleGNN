#!/bin/sh
DATA_PATH="/home/mlai21/seiffart/Data/GraphData/DS_all/"
DATABASE_NAME="DHFR"
MAX_RUN_ID=2
CROSS_VALIDATION=9
NETWORK_TYPES="100;wl_1:1,2,3;wl_1 200;wl_1:1,2,3;wl_1 500;wl_1:1,2,3;wl_1 100;wl_2:1,2,3;wl_2 200;wl_2:1,2,3;wl_2 500;wl_2:1,2,3;wl_2 100;wl_1:1,2,3,4,5,6;wl_1 100;wl_2:1,2,3,4,5,6;wl_2 500;wl_2:1,2,3,4,5,6;wl_2 100;wl_1:1,2,3;wl_1:1,2,3;wl_1 500;wl_2:1,2,3;wl_2:1,2,3;wl_2"
#NETWORK_TYPES="100;wl_2:1,2,3;wl_2 200;wl_2:1,2,3;wl_2 500;wl_2:1,2,3;wl_2"
EPOCHS=50
BATCH_SIZE=16
EDGE_LABELS=-1
USE_FEATURES=True
LEARNING_RATE=0.001
BALANCED=False
LOAD_SPLITS=True
CONVOLUTION_GRAD=True
RESIZE_GRAD=True

for network_type in $NETWORK_TYPES; do
  (for i in $(seq 0 $MAX_RUN_ID); do
      for j in $(seq 0 $CROSS_VALIDATION); do
        python GraphLearningMain.py --data_path $DATA_PATH --graph_db_name $DATABASE_NAME --run_id $i --validation_id $j --network_type $network_type --epochs $EPOCHS --batch_size $BATCH_SIZE --edge_labels $EDGE_LABELS --use_features $USE_FEATURES --lr $LEARNING_RATE --load_splits $LOAD_SPLITS  --balanced $BALANCED --convolution_grad $CONVOLUTION_GRAD --resize_grad $RESIZE_GRAD --mode fast &
      done
  done
  wait)
done

