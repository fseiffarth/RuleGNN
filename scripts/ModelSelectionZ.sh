#!/bin/sh
DATABASE_NAME="ZINC"
CROSS_VALIDATION=0
CONFIG_FILES="Configs/Test/config_ZINC_1.yml Configs/Test/config_ZINC_2.yml"
export OMP_NUM_THREADS=1
for CONFIG_FILE in $CONFIG_FILES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config $CONFIG_FILE &
  done
done

