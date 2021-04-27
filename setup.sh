#!/bin/bash

mkdir thirdparty
cd thirdparty

echo Installing suite2p...
git clone git@github.com:MouseLand/suite2p.git
cd suite2p
conda env create -f environment.yml -p ./

echo Fetching ast_model...
cd ../
git clone git@github.com:znamlab/ast_model.git
