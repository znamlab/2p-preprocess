#!/bin/bash
#
#SBATCH --job-name=2p-preprocess-gpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/users/znamenp/code/2p-preprocess/logs/2p_preprocess_%j.log
ml purge
ml CUDA/10.1.105
ml cuDNN/7.5.0.56-CUDA-10.1.105
ml Anaconda3/2020.07

source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
cd /camp/home/znamenp/home/users/znamenp/code/2p-preprocess
preprocess2p ${PROJECT} ${SESSION} -c append --run-neuropil -t ${TAU}
