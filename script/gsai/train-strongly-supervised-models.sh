#!/usr/bin/env bash

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

# SS_MODEL_LEARNING_DIR_PATH=./model-instance/strongly-supervised-models
SS_MODEL_LEARNING_DIR_PATH=./model-instance/strongly-supervised-models

python -m domain.kqapro.run --using-tqdm false \
       --config config.train_for_multiple_decoding_strategies \
       --model-learning-dir $SS_MODEL_LEARNING_DIR_PATH \
       --additional-config config.additional.weaksup_pretraining

# PERCENT=0.1  # between 0 to 100
# python -m domain.kqapro.run --using-tqdm false \
#        --config config.train_for_multiple_decoding_strategies \
#        --model-learning-dir $SS_MODEL_LEARNING_DIR_PATH \
#        --additional-config config.additional.train_set_portion \
#        --train-set-percent $PERCENT --epoch-repeat-strategy sqrt
