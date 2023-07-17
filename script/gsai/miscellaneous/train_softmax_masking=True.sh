#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --using-tqdm false --config config.train_strong_sup --additional-config config.additional.softmax_masking=True
