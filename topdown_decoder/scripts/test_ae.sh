#!/bin/bash --login
#$ -cwd             # Job will run from the current directory
                    # NO -V line - we load modulefiles in the jobscript
#$ -l a100=1 # We request only 1 GPU
#$ -pe smp.pe 8 # We want a CPU core for each process (see below)
module load apps/binapps/anaconda3/2021.11  # Python 3.9.7
module load libs/cuda

source activate diffuse_vae

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/lighting_5_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type Lighting \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/lighting_5_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type Lighting \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/defocus_blur_5_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Defocus Blur" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/defocus_blur_5_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Defocus Blur" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# # Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/spatter_3_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Spatter" \
                                --severity 3 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/spatter_3_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Spatter" \
                                --severity 3 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/speckle_noise_3_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Speckle Noise" \
                                --severity 3 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/speckle_noise_3_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Speckle Noise" \
                                --severity 3 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test


# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/ae_test/ \
                                --write-mode image \
                                --adapt None \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test


# Gibson: Trained with the frozen Visual Encoder from Habitat training
HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test


HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/motion_blur_5_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Motion Blur" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test


HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/fog_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Fog" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/rain_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Rain" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/snow_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Snow" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/shadow_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Shadow" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/glare_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Glare" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/jitter_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Jitter" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/occlusion_ae_dua_test/ \
                                --write-mode image \
                                --adapt dua \
                                --corruption-type "Occlusion" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/motion_blur_5_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Motion Blur" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test


HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/fog_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Fog" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/rain_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Rain" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/snow_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Snow" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/shadow_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Shadow" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/glare_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Glare" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/jitter_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Jitter" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test

HYDRA_FULL_ERROR=1 CUDA_VISIBLE_DEVICE=0 python main/test_ae.py reconstruct --device gpu:0 \
                                --dataset gibson \
                                --image-size 256 \
                                --num-samples 20 \
                                --save-path outputs/occlusion_ae_test/ \
                                --write-mode image \
                                --adapt None \
                                --corruption-type "Occlusion" \
                                --severity 5 \
                                logs/ae_gibson/checkpoints/ae_gibson.ckpt \
                                datasets/gibson_test