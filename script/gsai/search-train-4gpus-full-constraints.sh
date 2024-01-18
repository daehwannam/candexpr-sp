#!/usr/bin/env bash

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

export NCCL_P2P_LEVEL=NVL

export CUDA_VISIBLE_DEVICES=$(seq -s , 0 3)  # 0,1,2,3
NUM_GPUS=$(($(echo $CUDA_VISIBLE_DEVICES | tr -cd , | wc -c) + 1))  # 4
ACCELERATE_CONFIG="accelerate/${NUM_GPUS}gpus.yaml"

DECODING=full-constraints
EXTRA_OPTIONS=''

SS_MODEL_LEARNING_DIR_PATH=./model-instance-keep/20240112-strongly-supervised-models
COMMON_WS_MODEL_LEARNING_DIR_PATH=./model-instance/20240112-weakly-supervised-models

PRETRAINED_MODEL_PATH=$SS_MODEL_LEARNING_DIR_PATH/$DECODING:best/model
WS_MODEL_LEARNING_DIR_PATH=$COMMON_WS_MODEL_LEARNING_DIR_PATH/$DECODING

accelerate launch --num_processes $NUM_GPUS --config_file $ACCELERATE_CONFIG \
           --main_process_port $(shuf -i 49152-65535 -n 1) \
           -m domain.kqapro.run --using-tqdm false \
           --config config.search_train \
           --model-learning-dir $WS_MODEL_LEARNING_DIR_PATH \
           --pretrained-model-path $PRETRAINED_MODEL_PATH \
           --resuming false \
           $EXTRA_OPTIONS
