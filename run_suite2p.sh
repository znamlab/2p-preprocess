#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
ml purge
ml CUDA/11.3.1
ml cuDNN/8.2.1.32-CUDA-11.3.1
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c append --run-neuropil y -t ${TAU} --run-split y
