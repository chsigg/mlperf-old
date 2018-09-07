#!/bin/bash
# This script should be only executed in docker.
# Run minigo... stop when it converges.
set -e

SEED=$1
cd minigo
bazel run -c opt --define=tf=1 --define=board_size=9 \
    //:loop_main -- params/final.json $SEED
