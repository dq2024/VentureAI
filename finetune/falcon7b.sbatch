#!/bin/bash

#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=falcon7b
#SBATCH --mail-user=dq2024@nyu.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=falcon7b.out
#SBATCH --gres=gpu:0


module purge
cd /scratch/$USER
singularity exec --nv\
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:r \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif\
    /bin/bash -c "source /ext3/env.sh; conda activate venture_ai_env; cd /home/$USER/VentureAI/finetune; torchrun --nproc_per_node=1 falcon7b.py"