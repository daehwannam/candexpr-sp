#!/usr/bin/sh

set -e

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

MODULE=semparse_baseline
PROCESSED_PATH=./baseline-processed/bart-program

CHECKPOINT_PATHS=./output-keep/bart-program-*/checkpoint-best

mkdir -p ./output-test

for checkpoint_path in $CHECKPOINT_PATHS; do
    model_name=$(basename $(dirname $checkpoint_path))
    output_dir_path=./output-test/${model_name}
    # echo $output_dir_path
    python -m $MODULE.predict --input_dir $PROCESSED_PATH --save_dir  $output_dir_path --ckpt $checkpoint_path --postprocessing-answer
done
