#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --using-tqdm false --config config.train_for_multiple_decoding_strategies \
       --additional-config 'config.batch.size=128_and_num_epochs=100'
