#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

MODULE=semparse_baseline
DATE=$(date '+%Y-%m-%d_%H:%M:%S')

# MODEL_NAME_OR_PATH='facebook/bart-base'
MODEL_NAME_OR_PATH=./pretrained/bart-base
PROCESSED_PATH=./baseline-processed/bart-program
OUTPUT_DIR_PATH=./output/bart-program-$DATE
TRAIN_LOG_PATH=./log/bart-train-program-$DATE
# PREDICT_LOG_PATH=./log/bart-predict-program-$DATE

BATCH_SIZE=128
NUM_TRAIN_EPOCHS=100
MAX_GRAD_NORM=0.1

python -m $MODULE.train --input_dir $PROCESSED_PATH --output_dir $OUTPUT_DIR_PATH --save_dir $TRAIN_LOG_PATH --model_name_or_path "$MODEL_NAME_OR_PATH" --postprocessing-answer --use-shuffled-train --disable-progress-bar --batch_size $BATCH_SIZE --num_train_epochs $NUM_TRAIN_EPOCHS --max_grad_norm $MAX_GRAD_NORM
