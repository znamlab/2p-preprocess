#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
ml purge
ml CUDA/12.1.1
ml cuDNN/8.9.2.26-CUDA-12.1.1

source ${UV_PROJECT_ENVIRONMENT}/bin/activate

echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c append --run-neuropil  -t ${TAU} --run-split 
