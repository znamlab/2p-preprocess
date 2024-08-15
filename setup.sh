#!/bin/bash

echo Install dependencies...
conda env create -f environment.yml

conda activate 2p-preprocess
conda install pip
pip install -e .
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda deactivate
