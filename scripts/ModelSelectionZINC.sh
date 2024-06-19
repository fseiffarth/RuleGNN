#!/bin/sh
DATABASE_NAME="ZINC"
CROSS_VALIDATION=0
#CONFIG_FILES="config_ZINC1.yml config_ZINC2.yml config_ZINC3.yml config_ZINC4.yml config_ZINC5.yml config_ZINC6.yml config_ZINC7.yml"
# ZINC_best1 to ZINC_best15
CONFIG_FILES="config_ZINC_best1.yml config_ZINC_best2.yml config_ZINC_best3.yml config_ZINC_best4.yml config_ZINC_best5.yml config_ZINC_best6.yml config_ZINC_best7.yml config_ZINC_best8.yml config_ZINC_best9.yml config_ZINC_best10.yml config_ZINC_best11.yml config_ZINC_best12.yml config_ZINC_best13.yml config_ZINC_best14.yml config_ZINC_best15.yml"
for CONFIG_FILE in $CONFIG_FILES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config 'Configs/ZINC_original/'$CONFIG_FILE &
  done
done

