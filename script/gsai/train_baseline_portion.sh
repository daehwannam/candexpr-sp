#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

PERCENT=$1

MODULE=semparse_baseline
DATE=$(date '+%Y-%m-%d_%H:%M:%S')

# MODEL_NAME_OR_PATH='facebook/bart-base'
MODEL_NAME_OR_PATH=./pretrained/bart-base
PROCESSED_PATH=./baseline-processed/bart-program
OUTPUT_DIR_PATH=./output/bart-program-$DATE
TRAIN_LOG_PATH=./log/bart-train-program-$DATE
PREDICT_LOG_PATH=./log/bart-predict-program-$DATE

python -m $MODULE.train --input_dir $PROCESSED_PATH --output_dir $OUTPUT_DIR_PATH --save_dir $TRAIN_LOG_PATH --model_name_or_path "$MODEL_NAME_OR_PATH" --postprocessing-answer --train-set-percent $PERCENT --use-shuffled-train --disable-progress-bar
