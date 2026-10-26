#!/bin/bash --login
#
#SBATCH --job-name=2p-preprocess-dff-noneuropil
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=256G
#SBATCH --partition=ncpu
#SBATCH --mail-type=END,FAIL
conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
2p calcium -p ${PROJECT} -s ${SESSION} -c ${CONFLICTS} --no-run-suite2p --no-run-neuropil  -t ${TAU} --run-split
