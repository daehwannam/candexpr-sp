#!/usr/bin/sh

set -e

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

BASE_DIR='model-instance-keep/2023-07-19_02:25:06_534864_batch_size=16_percent=100'
DIRS="$BASE_DIR/*:best"

# for dir in $DIRS; do
#     # python -m domain.kqapro.run --config config.test_on_test_set --model-checkpoint-dir $dir --test-dir-name $dir/test-on-test-set
#     python -m domain.kqapro.run --config config.test_on_val_set --model-checkpoint-dir $dir --test-dir-name $dir/test-on-val-set
# done

dir="$BASE_DIR/full-constraints:best"
python -m domain.kqapro.run --config config.test_on_val_set --model-checkpoint-dir $dir --test-dir-name $dir/test-on-val-set

dir="$BASE_DIR/no-arg-candidate:best"
python -m domain.kqapro.run --config config.test_on_val_set --model-checkpoint-dir $dir --test-dir-name $dir/test-on-val-set --extra-config config.additional.using_arg_candidate=False

dir="$BASE_DIR/no-constrained-decoding:best"
python -m domain.kqapro.run --config config.test_on_val_set --model-checkpoint-dir $dir --test-dir-name $dir/test-on-val-set --extra-config config.additional.constrained_decoding=False
