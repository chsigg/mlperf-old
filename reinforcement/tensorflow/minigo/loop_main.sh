#!/bin/bash

set -e

export GOPARAMS=$1
SEED=$2

FILE="TERMINATE_FLAG"
rm -f $FILE

./loop_init 2>&1

for i in {1..1000}; do

  ./loop_selfplay $SEED $i 2>&1
  ./loop_train_eval $SEED $i 2>&1

  if [ -f $FILE ]; then
    echo "$FILE exists: finished!"
    cat $FILE
    break
  else
    echo "$FILE does not exist; looping again."
  fi

done
