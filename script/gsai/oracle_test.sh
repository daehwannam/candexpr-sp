
# LEARNING_DIRS='
#     ./model-instance-keep/multiple_decoding_strategies__0.1
#     ./model-instance-keep/multiple_decoding_strategies__0.3
#     ./model-instance-keep/multiple_decoding_strategies__1
#     ./model-instance-keep/multiple_decoding_strategies__10
#     ./model-instance-keep/multiple_decoding_strategies__100
#     ./model-instance-keep/multiple_decoding_strategies__3
#     ./model-instance-keep/multiple_decoding_strategies__30'

LEARNING_DIR='model-instance-keep/20230821/multiple_decoding_strategies__0.1'
TEST_CONFIG='config.oracle_test_on_val_set'
# SBATCH_SCRIPT=~/sbatch/3090
SBATCH_SCRIPT=sbatch/gsai/3090.sh


for BEAM_SIZE in '4'; do
# for BEAM_SIZE in '4 8 12 16'; do
    BEAM_CONFIG="config.extra.num_prediction_beams=$BEAM_SIZE|config.extra.test_batch_size=4"

    CHECKPOINT_PATH="${LEARNING_DIR}/full-constraints:best"
    TEST_DIR_PATH="model-test-keep/20231122-oracle/beam-${BEAM_SIZE}/full-constraints"
    EXTRA_CONFIG="${BEAM_CONFIG}"
    TEST_CMD="python -m domain.kqapro.run --config config.oracle_test_on_val_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --extra-config $EXTRA_CONFIG"
    sbatch $SBATCH_SCRIPT $TEST_CMD

    CHECKPOINT_PATH="${LEARNING_DIR}/no-arg-candidate:best"
    TEST_DIR_PATH="model-test-keep/20231122-oracle/beam-${BEAM_SIZE}/no-arg-candidate"
    EXTRA_CONFIG="${BEAM_CONFIG}|config.extra.using_arg_candidate=False"
    TEST_CMD="python -m domain.kqapro.run --config config.oracle_test_on_val_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --extra-config $EXTRA_CONFIG"
    sbatch $SBATCH_SCRIPT $TEST_CMD

    CHECKPOINT_PATH="${LEARNING_DIR}/no-ac-no-dut:best"
    TEST_DIR_PATH="model-test-keep/20231122-oracle/beam-${BEAM_SIZE}/no-ac-no-dut"
    EXTRA_CONFIG="${BEAM_CONFIG}|config.extra.using_arg_candidate=False|config.extra.using_distinctive_union_types=False"
    TEST_CMD="python -m domain.kqapro.run --config config.oracle_test_on_val_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --extra-config $EXTRA_CONFIG"
    sbatch $SBATCH_SCRIPT $TEST_CMD

    CHECKPOINT_PATH="${LEARNING_DIR}/no-constrained-decoding:best"
    TEST_DIR_PATH="model-test-keep/20231122-oracle/beam-${BEAM_SIZE}/no-constrained-decoding"
    EXTRA_CONFIG="${BEAM_CONFIG}|config.extra.constrained_decoding=False"
    TEST_CMD="python -m domain.kqapro.run --config config.oracle_test_on_val_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir $TEST_DIR_PATH --extra-config $EXTRA_CONFIG"
    sbatch $SBATCH_SCRIPT $TEST_CMD
done
