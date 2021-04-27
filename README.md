# Installing the pipeline

Clone the repo from github:
```
git clone git@github.com:znamlab/2p-preprocess.git
```

Next, navigate to the repo directory and run the setup script:
```
cd 2p-preprocess
./setup.sh
```

This should install the dependencies and create a virtual enviroment for suite2p.

To start the slurm job, navigate to the `2p-preprocess` directory and run the following:
```
sbatch --export=DATA=<DIRECTORY-WITH-YOUR-TIFFS> run_suite2p.sh
```
