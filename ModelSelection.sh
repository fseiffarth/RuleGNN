#!/bin/sh
DATABASE_NAMES="IMDB-BINARY"
CROSS_VALIDATION=1
CONFIG_FILE="Configs/config_IMDB-BINARY_all_speed.yml"
export MKL_NUM_THREADS=3
export NUMEXPR_NUM_THREADS=3
export OMP_NUM_THREADS=3
for DATABASE_NAME in $DATABASE_NAMES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config $CONFIG_FILE &
  done
done

