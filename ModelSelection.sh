#!/bin/sh
DATABASE_NAMES="NCI1 NCI109 Mutagenicity"
CROSS_VALIDATION=10
CONFIG_FILE="config_NCI1.yml"
for DATABASE_NAME in $DATABASE_NAMES; do
  for j in $(seq 1 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --validation_number $CROSS_VALIDATION --config $CONFIG_FILE &
  done
done

