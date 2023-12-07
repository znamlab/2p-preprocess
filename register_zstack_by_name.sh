#!/bin/bash
#
#SBATCH --job-name=register_zstack
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

2p zstack -p ccyp_ex-vivo-reg-pilot -s $SESSION --conflicts skip -c 0 --max-shift 50 --dataset_name $STACK_NAME
