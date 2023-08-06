#!/usr/bin/sh

set -e

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

TRAIN_DIR_PATH=$1
TRAIN_DIR_NAME=$(basename $TRAIN_DIR_PATH)
# e.g. TRAIN_DIR_PATH=./model-instance-keep/2023-07-19_02:25:06_534864_batch_size=16_percent=100

# CHECKPOINT_DIR_NAMES='
# full-constraints:best
# no-arg-candidate:best
# no-constrained-decoding:best
# '

# for checkpoint_dir_name in $CHECKPOINT_DIR_NAMES; do
#     CHECKPOINT_DIR_PATH=${TRAIN_DIR_PATH}/${checkpoint_dir_name}
#     echo $CHECKPOINT_DIR_PATH
# done

# TEST_CONFIG='test_on_test_set'

ADDITIONAL_CONFIG_COMMON='config.additional.num_prediction_beams=4|config.additional.test_batch_size=16'
# ADDITIONAL_CONFIG_COMMON='config.additional.num_prediction_beams=4'

# for test_config in 'test val'; do
for data_type in 'val'; do
    TEST_CONFIG="config.test_on_${data_type}_set"

    CHECKPOINT_DIR_NAME='full-constraints:best'
    CHECKPOINT_DIR_PATH=${TRAIN_DIR_PATH}/${CHECKPOINT_DIR_NAME}
    TEST_DIR_PATH=./model-test-keep/${TRAIN_DIR_NAME}:on-${data_type}:b4_b16:${CHECKPOINT_DIR_NAME}
    ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG_COMMON}"
    python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_DIR_PATH --test-dir $TEST_DIR_PATH --additional-config $ADDITIONAL_CONFIG

    CHECKPOINT_DIR_NAME='no-arg-candidate:best'
    CHECKPOINT_DIR_PATH=${TRAIN_DIR_PATH}/${CHECKPOINT_DIR_NAME}
    TEST_DIR_PATH=./model-test-keep/${TRAIN_DIR_NAME}:on-${data_type}:b4_b16:${CHECKPOINT_DIR_NAME}
    ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG_COMMON}|config.additional.using_arg_candidate=False"
    python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_DIR_PATH --test-dir $TEST_DIR_PATH --additional-config $ADDITIONAL_CONFIG

    CHECKPOINT_DIR_NAME='no-ac-no-dut:best'
    CHECKPOINT_DIR_PATH=${TRAIN_DIR_PATH}/${CHECKPOINT_DIR_NAME}
    TEST_DIR_PATH=./model-test-keep/${TRAIN_DIR_NAME}:on-${data_type}:b4_b16:${CHECKPOINT_DIR_NAME}
    ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG_COMMON}|config.additional.using_arg_candidate=False|config.additional.using_distinctive_union_types=False"
    python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_DIR_PATH --test-dir $TEST_DIR_PATH --additional-config $ADDITIONAL_CONFIG

    CHECKPOINT_DIR_NAME='no-constrained-decoding:best'
    CHECKPOINT_DIR_PATH=${TRAIN_DIR_PATH}/${CHECKPOINT_DIR_NAME}
    TEST_DIR_PATH=./model-test-keep/${TRAIN_DIR_NAME}:on-${data_type}:b4_b16:${CHECKPOINT_DIR_NAME}
    ADDITIONAL_CONFIG="${ADDITIONAL_CONFIG_COMMON}|config.additional.constrained_decoding=False"
    python -m domain.kqapro.run --using-tqdm false --config $TEST_CONFIG --model-checkpoint-dir $CHECKPOINT_DIR_PATH --test-dir $TEST_DIR_PATH --additional-config $ADDITIONAL_CONFIG
done
