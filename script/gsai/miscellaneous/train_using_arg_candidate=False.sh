#!/usr/bin/sh

source /home/dhnam/program/miniconda3/etc/profile.d/conda.sh
conda activate kqapro

python -m domain.kqapro.run --using-tqdm false --config config.train_strong_sup --extra-config config.extra.using_arg_candidate=False
