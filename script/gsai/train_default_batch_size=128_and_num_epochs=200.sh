#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --using-tqdm false --config config.train_default \
       --model-learning-dir './model-instance/train_default__batch_size=128__num_epochs=200' \
       --additional-config 'config.batch.size=128_and_num_epochs=200'
