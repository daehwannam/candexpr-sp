#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --using-tqdm false --config config.train_for_multiple_decoding_strategies \
       --model-learning-dir './model-instance/multiple_decoding_strategies__batch_size=128__num_epochs=100' \
       --additional-config 'config.batch.size=128_and_num_epochs=100'
