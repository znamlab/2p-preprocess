#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/users/znamenp/code/2p-preprocess/logs/2p_preprocess_%j.log
ml purge

ml Anaconda3/2020.07
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate 2p-preprocess
echo Processing ${SESSION} from ${MOUSE} in project ${PROJECT}
cd /camp/home/znamenp/home/users/znamenp/code/2p-preprocess
preprocess2p ${PROJECT} ${MOUSE} ${SESSION} -c skip --run-neuropil
