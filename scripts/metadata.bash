#/bin/bash
#
# 1. Audio and VAD Directories
dataset="Travel"
audio_dir="audio"

for dataset in CallfullSPA CallfullDEU;
do
    
    python vap_datasets/datamodule/create_metadata.py --dataset $dataset

done