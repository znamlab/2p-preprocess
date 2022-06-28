#!/bin/bash
#SBATCH --job-name=2p-preprocess
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --partition=hmem
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/users/hey2/codes/2p-preprocess/logs/2p_preprocess_%j.log
ml purge

ml Anaconda3/2020.07
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate 2p-preprocess
echo Processing ${SESSION} in project ${PROJECT}
cd /camp/lab/znamenskiyp/home/users/hey2/codes/2p-preprocess/
preprocess2p ${PROJECT} ${SESSION} -c append -t ${TAU}