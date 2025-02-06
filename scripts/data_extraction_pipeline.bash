#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=10:00:00
#PJM -L jobenv=singularity
#PJM -j

eval "$(/home/z40351r/anaconda3/bin/conda shell.bash hook)"
conda activate VAP

# 1. Audio and VAD Directories
dataset_name="CEJC"
audio_dir="audio"
label_dir="vadTravelbert50" # original or webrtc or vad{exp_dir}{threshold}

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


# # 1. Create Overview
# python vap_datasets/datamodule/create_overview_csv.py --audio_dir $AUDIO_DIR --vad_dir $LABEL_DIR --output $OVERVIEW_CSV

# python vap_datasets/datamodule/create_metadata.py --dataset $dataset_name

# 2. Create Splits
# Callfull, Callfriend, MDTには使用禁止！
# python vap_datasets/datamodule/create_splits.py --csv $OVERVIEW_CSV --output_dir $SPLIT_DIR

# 二回目以降はここから
# 3. Create path list
# python3 vap_datasets/datamodule/create_subset_csv.py --dataset $dataset_name --audio_dir $AUDIO_DIR --label_dir $LABEL_DIR --out_dir $PATH_DIR

# # # # 4. Create Data (TRAIN/VAL)
# python vap_datasets/datamodule/create_sliding_window_dset.py --audio_vad_csv $PATH_DIR/train.csv --output $TRAIN
# python vap_datasets/datamodule/create_sliding_window_dset.py --audio_vad_csv $PATH_DIR/val.csv --output $VALIDATION
# python vap_datasets/datamodule/create_sliding_window_dset.py --audio_vad_csv $PATH_DIR/test.csv --output $TEST

# # 5. Create Classification Data
# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/val.csv --output $VAL_CLASSIFICATION
# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/train.csv --output $TRAIN_CLASSIFICATION
# python vap_datasets/datamodule/dset_event.py --audio_vad_csv $PATH_DIR/test.csv --output $TEST_CLASSIFICATION

# # 6. option
# python vap_datasets/datamodule/shift_gap.py --dataset ${dataset_name} --out_dir ${out_dir}
python vap_datasets/eval/eval_label.py --dataset ${dataset_name} --label_name ${label_dir}

# # # 6. Option create label distribution
# python vap_datasets/labels/labels.py \
#     datamodule.datasets=\[$dataset_name\] \
#     datamodule.subsets=\[$out_dir\] \