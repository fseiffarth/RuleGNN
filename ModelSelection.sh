#!/bin/sh
DATABASE_NAMES="NCI1 NCI109 Mutagenicity"
CROSS_VALIDATION=9
CONFIG_FILE="config_NCI1.yml"
for DATABASE_NAME in $DATABASE_NAMES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config $CONFIG_FILE &
  done
done

