#!/bin/bash

echo Install dependencies...
conda env create -f environment.yml

conda activate 2p-preprocess
pip install -e .
conda deactivate
