#!/bin/bash
#
#SBATCH --job-name=scvae
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=suite2p_%j.log
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

conda activate /camp/lab/znamenskiyp/home/shared/code/suite2p

cd /camp/lab/znamenskiyp/home/shared/code/2p-preprocess
python preprocess_2p.py ~/${DATA}

ml matlab/R2019a
