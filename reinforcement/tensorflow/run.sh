#!/bin/bash
# This script should be only executed in docker.
# Run minigo... stop when it converges.
set -e

SEED=$1
cd minigo
bazel run -c opt //:loop_main -- params/final.json $SEED
