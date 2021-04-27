#!/bin/bash
#
#SBATCH --job-name=2p-preprocesss
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --partition=cpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/camp/lab/znamenskiyp/home/shared/code/2p-preprocess/logs/2p_preprocess_%j.log
ml purge
ml Anaconda3
source /camp/apps/eb/software/Anaconda/conda.env.sh

cd ${0%/*}
conda activate ./thirdparty/suite2p
python preprocess_2p.py ${DATA}

ml matlab/R2019a
