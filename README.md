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

This should install the dependencies and create conda environments for suite2p
and for the repo itself. Environments are created in each users home directory.

`run_suite2p.sh` contains an example script that first runs the standard run_suite2p
pipeline and then applies neuropil correction using the AST model.

To start the slurm job, navigate to the `2p-preprocess` directory and run the following:
```
sbatch --export=DATA=<DIRECTORY-WITH-YOUR-TIFFS> run_suite2p.sh
```

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
