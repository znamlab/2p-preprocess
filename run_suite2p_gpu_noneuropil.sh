#!/bin/bash
#
#SBATCH --job-name=2p-preprocess-gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=ga100
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
ml purge
ml CUDA/12.1.1
ml cuDNN/8.9.2.26-CUDA-12.1.1
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c ${CONFLICTS} --no-run-neuropil  -t ${TAU} --run-split --run-suite2p
