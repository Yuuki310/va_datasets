#!/bin/bash

#PJM -L rscgrp=cx-share
#PJM -L gpu=1
#PJM -L elapse=1:00:00
#PJM -L jobenv=singularity
#PJM -j

module load singularity
singularity exec \
	--bind $HOME,/data/group1/z40351r \
	--nv /data/group1/z40351r/my_container.sif \
	bash datasets_turntaking/datasets/Callhome/upsampling.sh