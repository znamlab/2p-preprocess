#!/bin/bash
#
#SBATCH --job-name=2p-preprocess-gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/users/hey2/codes/duplicates/2p-preprocess/logs/2p_preprocess_%j.log
ml purge
ml CUDA/11.3.1
ml cuDNN/8.2.1.32-CUDA-11.3.1
ml Anaconda3/2020.07

source activate base

conda activate 2p-preprocess_test2
echo Processing ${SESSION} in project ${PROJECT}
cd /camp/lab/znamenskiyp/home/users/hey2/codes/duplicates/2p-preprocess
preprocess2p ${PROJECT} ${SESSION} -c append --run-neuropil -t ${TAU}
preprocess2p ${PROJECT} ${SESSION} -c skip -t ${TAU} --run-split