#/bin/bash
#
# 1. Audio and VAD Directories
dataset_name="CEJC"


 python vap_datasets/webrtc/make_vad.py --dataset $dataset_name --out_dir "webrtc05"

 