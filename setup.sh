#!/bin/bash

mkdir thirdparty
cd thirdparty

echo Installing suite2p...
git clone git@github.com:MouseLand/suite2p.git
conda env create -f suite2p/environment.yml

echo Install dependencies...
conda env create -f environment.yml

echo Fetching ast_model...
cd ../
git clone git@github.com:znamlab/ast_model.git
