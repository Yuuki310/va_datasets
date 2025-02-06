#/bin/bash
#
# 1. Audio and VAD Directories
dataset_name="CEJC"
subset="vad_CEJC_bert1s3c50"
# out_dirがdefaultかwebrtcしかないので
# 音源の種類が増えたら考える
if [ "$vad_dir" = "vad" ]; then
    out_dir="default"
else
    out_dir="$vad_dir"
fi



python3 vap_datasets/plot/plot_dataset.py --dataset $dataset_name --subset $subset