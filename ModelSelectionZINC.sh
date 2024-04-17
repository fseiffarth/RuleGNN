#!/bin/sh
DATABASE_NAME="ZINC"
CROSS_VALIDATION=0
CONFIG_FILES="config_ZINC1.yml config_ZINC2.yml config_ZINC3.yml config_ZINC4.yml config_ZINC5.yml config_ZINC6.yml config_ZINC7.yml"

for CONFIG_FILE in $CONFIG_FILES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config $CONFIG_FILE &
  done
done

