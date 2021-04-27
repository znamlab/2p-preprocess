# Installing suite2p

Clone suite2p from github:
```
git clone git@github.com:MouseLand/suite2p.git
```

Next, navigate to the suite2p directory and create the conda environment at the
current location:
```
conda env create -f environment.yml -p ./
```

To start the slurm job, navigate back to the `2p-preprocess` directory and run the following:
```
sbatch --export=DATA=<DIRECTORY-WITH-YOUR-TIFFS> run_suite2p.sh
```
