#/bin/bash
#
# 1. Audio and VAD Directories
dataset_name="Travel"
label_name="vadTravelbert50"


python vap_datasets/eval/eval_label.py \
    --dataset $dataset_name --label_name $label_name
