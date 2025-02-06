#/bin/bash
#
# 1. Audio and VAD Directories
module load singularity
singularity exec \
	--bind $HOME,/data/group1/z40351r \
	--nv /data/group1/z40351r/my_container.sif \
    bash run.sh storage_dir
    
python vap_datasets/datasets/HKUST/prepare_data.py
