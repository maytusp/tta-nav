#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
                    # NO -V line - we load modulefiles in the jobscript
#$ -l a100=1 # We request only 1 GPU
#$ -pe smp.pe 8 # We want a CPU core for each process (see below)
module load apps/binapps/anaconda3/2020.07  # Python 3.8.3
module load libs/cuda/11.0.3

source activate diffuse_vae

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/train_ae.py +dataset=gibson/train \
                     dataset.vae.data.root='datasets/gibson_train' \
                     dataset.vae.data.name='gibson' \
                     dataset.vae.data.hflip=True \
                     dataset.vae.training.batch_size=128 \
                     dataset.vae.training.log_step=50 \
                     dataset.vae.training.epochs=200 \
                     dataset.vae.training.device=\'gpu:0\' \
                     dataset.vae.training.results_dir=\'logs/ae_gibson/\' \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'gibson_large\' \
                     dataset.vae.training.alpha=1.0 \
