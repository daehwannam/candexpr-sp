#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --config config.train_for_multiple_decoding_strategies \
       --additional-config 'config.batch_size=128_and_num_epochs=100'
