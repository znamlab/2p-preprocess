#!/bin/bash
#
#SBATCH --job-name=2p-preprocess-gpu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
ml purge
ml CUDA/12.1.1
ml cuDNN/8.9.2.26-CUDA-12.1.1
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c ${CONFLICTS} --run-suite2p n --run-dff y --run-split y -t ${TAU}