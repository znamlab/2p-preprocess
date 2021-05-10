#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/users/znamenp/code/2p-preprocess/logs/2p_preprocess_%j.log
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

cd /camp/lab/znamenskiyp/home/users/znamenp/code/2p-preprocess
conda activate suite2p
python preprocess_2p.py ${PROJECT} ${MOUSE} ${DATE} ${SESSION}

conda activate 2p-preprocess
neuropil ${DATA}/suite2p/plane0/
