#!/bin/bash
#SBATCH --nodes=1                        # requests 1 compute server
#SBATCH --ntasks-per-node=1              # runs 1 tasks on each server
#SBATCH --cpus-per-task=4                # uses 4 compute cores per task
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64GB
#SBATCH --job-name=falcon7b
#SBATCH --output=falcon7b.out
#SBATCH --error=falcon7b-error.txt
#SBATCH --mail-user=dq2024@nyu.edu

cd /scratch/$USER
singularity exec --nv\
    --overlay /scratch/$USER/overlay-25GB-500K.ext3:rw \
    /scratch/work/public/singularity/cuda11.3.0-cudnn8-devel-ubuntu20.04.sif\
    /bin/bash -c "source /ext3/env.sh; conda activate venture_ai_env; cd /home/$USER/VentureAI/finetune; torchrun --nproc_per_node=1 falcon7b.py"