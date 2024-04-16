#!/bin/sh
DATABASE_NAMES="IMDB-BINARY IMDB-MULTI"
MAX_RUN_ID=2
CROSS_VALIDATION=9
CONFIG_FILE="config_IMDB-BINARY.yml"

for DATABASE_NAME in $DATABASE_NAMES; do
  (
  for i in $(seq 0 $MAX_RUN_ID); do
        for j in $(seq 0 $CROSS_VALIDATION); do
          python RunBestModel.py --graph_db_name $DATABASE_NAME --run_id $i --validation_id $j --config $CONFIG_FILE &
        done
  done
  wait
  )
done

