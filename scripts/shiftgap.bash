#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate VAP

# 1. Audio and VAD Directories
dataset_name="Switchboard"
audio_dir="audio"
label_dir="webrtc" # original or webrtc or vad_{exp_dir}{threshold}

# out_dirがdefaultかwebrtcしかないので
# 音源の種類が増えたら考える
out_dir="$label_dir"


DATA_DIR="data/"${dataset_name}
AUDIO_DIR="data/"${dataset_name}"/audio"
LABEL_DIR="data/"${dataset_name}"/labels/"${label_dir}
SPLIT_DIR="data/"${dataset_name}"/split"
PATH_DIR="data/"${dataset_name}"/path/"${out_dir}

OVERVIEW_CSV=${SPLIT_DIR}"/overview.csv"
TRAIN="data/"${dataset_name}"/path/"${out_dir}"/sliding_window_train.csv"
VALIDATION="data/"${dataset_name}"/path/"${out_dir}"/sliding_window_val.csv"
TEST="data/"${dataset_name}"/path/"${out_dir}"/sliding_window_test.csv"

TEST_CLASSIFICATION="data/"${dataset_name}"/path/"${out_dir}"/test_hs.csv"
VAL_CLASSIFICATION="data/"${dataset_name}"/path/"${out_dir}"/val_hs.csv"
TRAIN_CLASSIFICATION="data/"${dataset_name}"/path/"${out_dir}"/train_hs.csv"
ALL_CLASSIFICATION="data/"${dataset_name}"/path/"${out_dir}"/shifts.csv"

# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/val.csv --output $VAL_CLASSIFICATION
# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/train.csv --output $TRAIN_CLASSIFICATION
# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/test.csv --output $TEST_CLASSIFICATION

# # 6. option
python vap_datasets/datamodule/shift_gap.py --dataset ${dataset_name} --out_dir ${out_dir}
