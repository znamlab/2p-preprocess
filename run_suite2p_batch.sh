#!/bin/bash
#
#SBATCH --job-name=2p_analysis
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --mem=8G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/camp/lab/znamenskiyp/home/users/hey2/codes/2p-preprocess/logs/2p_preprocess_%j.log"

ml purge
ml Anaconda3/2022.05

source activate base

conda activate 2p-preprocess
cd "/camp/lab/znamenskiyp/home/users/hey2/codes/2p-preprocess/twop_preprocess/pipelines/"
echo Processing ${SESSION} in project ${PROJECT}
python preprocess_all_sessions.py