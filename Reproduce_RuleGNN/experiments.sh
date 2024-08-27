#!/bin/sh
DATABASE_NAMES="CSL EvenOddRings2_16 EvenOddRingsCount16 LongRings100 Snowflakes NCI1 NCI109 Mutagenicity DHFR IMDB-BINARY IMDB-MULTI"
CROSS_VALIDATIONS="4 9 9 9 9 9 9 9 9 9 9"
CONFIG_FILES="Reproduce_RuleGNN/Configs/config_CSL.yml Reproduce_RuleGNN/Configs/config_EvenOddRings.yml Reproduce_RuleGNN/Configs/config_EvenOddRingsCount.yml Reproduce_RuleGNN/Configs/config_LongRings.yml Reproduce_RuleGNN/Configs/config_Snowflakes.yml Reproduce_RuleGNN/Configs/config_NCI1.yml Reproduce_RuleGNN/Configs/config_NCI1.yml Reproduce_RuleGNN/Configs/config_NCI1.yml Reproduce_RuleGNN/Configs/config_DHFR.yml Reproduce_RuleGNN/Configs/config_IMDB.yml Reproduce_RuleGNN/Configs/config_IMDB.yml"
export OMP_NUM_THREADS=1
COUNTER=0
for DATABASE_NAME in $DATABASE_NAMES; do
  (
    for j in $(seq 0 ${CROSS_VALIDATIONS[$COUNTER]}); do
      python ModelSelection.py --graph_db_name $DATABASE_NAME --validation_id $j --config ${CONFIG_FILES[$COUNTER]} &
    done
    COUNTER=$((COUNTER+1))
    wait
  )
done