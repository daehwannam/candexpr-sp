#!/usr/bin/sh

set -e

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro


# for path in model-instance-keep/*; do
#     echo $(basename $path)
# done

DIRS='
2023-07-18_06:38:39_431530_batch_size=128_num_epochs=200
2023-07-19_01:17:04_094104_batch_size=128_num_epochs=100
2023-07-19_02:25:06_534864_batch_size=16_percent=100'

decodings='
full-constraints:best
no-arg-candidate:best
no-constrained-decoding:best
'

for dir in $DIRS; do
    for decoding in $decodings; do
        CHECKPOINT_PATH=model-instance-keep/$dir/$decoding
        TEST_DIR_NAME=${dir}_${decoding}
        # beam size = 1
        python -m domain.kqapro.run --config config.test_on_test_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir-name $TEST_DIR_NAME

        # beam size = 4
        python -m domain.kqapro.run --config config.test_on_test_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir-name $TEST_DIR_NAME --additional-config config.additional.num_prediction_beams=4
    done
done
