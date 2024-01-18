#!/usr/bin/bash

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

PERCENT=$1

python -m domain.kqapro.run --using-tqdm false --config config.train_for_multiple_decoding_strategies \
       --model-learning-dir ./model-instance/multiple_decoding_strategies__${PERCENT} \
       --extra-config config.additional.train_set_portion --train-set-percent $PERCENT
