#!/bin/bash

echo Install dependencies...
conda env create -f environment.yml

conda activate 2p-preprocess
conda install pip
pip install -e .
pip install --upgrade jax jaxlib==0.1.66+cuda101 -f https://storage.googleapis.com/jax-releases/jax_releases.html
conda deactivate
