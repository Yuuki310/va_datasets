#/bin/bash
#
# 1. Audio and VAD Directories
dataset_name="CEJC"
new_dataset=CEJC_phone

python vap_datasets/datamodule/copy_by_overview.py --dataset $dataset_name --new_dataset $new_dataset
