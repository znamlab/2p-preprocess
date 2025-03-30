#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/2p-preprocess/logs/2p_preprocess_%j.log"
ml purge
ml CUDA/11.3.1
ml cuDNN/8.2.1.32-CUDA-11.3.1
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess

echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c ${CONFLICTS} --run-suite2p n --run-dff y --run-split y -t ${TAU}