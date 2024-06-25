#!/bin/sh
DATABASE_NAME="ZINC"
CROSS_VALIDATION=0
CONFIG_FILES="Configs/Test/config_ZINC_1.yml Configs/Test/config_ZINC_2.yml Configs/Test/config_ZINC_3.yml Configs/Test/config_ZINC_4.yml Configs/Test/config_ZINC_5.yml Configs/Test/config_ZINC_6.yml Configs/Test/config_ZINC_7.yml Configs/Test/config_ZINC_8.yml Configs/Test/config_ZINC_9.yml Configs/Test/config_ZINC_10.yml Configs/Test/config_ZINC_11.yml Configs/Test/config_ZINC_12.yml Configs/Test/config_ZINC_13.yml Configs/Test/config_ZINC_14.yml Configs/Test/config_ZINC_15.yml"
export OMP_NUM_THREADS=1
for CONFIG_FILE in $CONFIG_FILES; do
  for j in $(seq 0 $CROSS_VALIDATION); do
    python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config $CONFIG_FILE &
  done
done

