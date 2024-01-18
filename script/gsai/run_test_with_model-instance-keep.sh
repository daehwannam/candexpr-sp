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
2023-07-19_02:25:06_534864_batch_size=16_percent=100

2023-07-19_02:23:37_863916_batch_size=16_percent=30
2023-07-19_02:25:10_428164_batch_size=16_percent=10
2023-07-19_02:25:38_635550_batch_size=16_percent=3
2023-07-19_02:26:06_414885_batch_size=16_percent=1
2023-07-19_02:26:16_951770_batch_size=16_percent=0.3
2023-07-19_02:26:19_489385_batch_size=16_percent=0.1
'

decodings='
full-constraints:best
no-arg-candidate:best
no-constrained-decoding:best
'

for dir in $DIRS; do
    for decoding in $decodings; do
        CHECKPOINT_PATH=model-instance-keep/$dir/$decoding
        TEST_DIR_NAME=${dir}_${decoding}

        if [ $decoding == 'no-arg-candidate:best' ]; then
            # echo $decoding
            extra_config='--extra-config config.extra.using_arg_candidate=False'
            extra_config_b4='--extra-config config.extra.using_arg_candidate=False_and_b=4'
            # echo $extra_config
        elif [ $decoding == 'no-constrained-decoding:best' ]; then
            # echo $decoding
            extra_config='--extra-config config.extra.constrained_decoding=False'
            extra_config_b4='--extra-config config.extra.constrained_decoding=False_and_b=4'
            # echo $extra_config
        else
            extra_config=''
            extra_config_b4='--extra-config config.extra.num_prediction_beams=4'
            # echo $decoding
        fi
        # beam size = 1
        python -m domain.kqapro.run --config config.test_on_test_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir-name $TEST_DIR_NAME $extra_config

        # beam size = 4
        python -m domain.kqapro.run --config config.test_on_test_set --model-checkpoint-dir $CHECKPOINT_PATH --test-dir-name $TEST_DIR_NAME $extra_config_b4
    done
done
