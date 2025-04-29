#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=gen_data_sp4
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=split4.out
#SBATCH --error=split4-error.txt

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

module purge

singularity exec ${nv} \
            --overlay /scratch/$USER/overlay-25GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
            /bin/bash -c "
source /ext3/env.sh; 
conda activate venture_ai_env; 
python3 generate_data_split_4.py
"