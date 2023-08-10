#!/usr/bin/sh

set -e

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

TEST_CONFIG='config.test_on_val_set'  # or 'config.test_on_test_set'
CHECKPOINT_PATH=$1
ACTION_NAME=$2
TEST_DIR_TAG=$3
TEST_DIR_PATH="./model-test-keep/test-on-val:${TEST_DIR_TAG}:no-arg-candidate:${ACTION_NAME}"



python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --additional-config config.additional.no_arg_candidate --no-arg-candidate-for $ACTION_NAME
# echo "python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --additional-config config.additional.no_arg_candidate --no-arg-candidate-for $ACTION_NAME"
