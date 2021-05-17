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

`run_suite2p.sh` and `run_suite2p_gpu.sh` contain example scripts that first runs the standard run_suite2p pipeline and then applies neuropil correction using the AST model.
If running neuropil correction using the AST model, using a GPU node is recommended.

To start the slurm job, navigate to the `2p-preprocess` directory and run the
`sbatch` script, passing the session details as environment variables, e.g.:
```
sbatch --export=PROJECT=test,MOUSE=PZAJ2.1c,SESSION=S20210513 run_suite2p_gpu.sh
```

# ASt model
The Asymmetric Student's t-model for neuropil correction is described [here](https://basellasermouse.github.io/ast_model/model.html). The python implementation
in this repository uses [JAX](https://github.com/google/jax) for automatic
differentiation and rapid GPU computation. If run on a node without GPU, it
should revert to CPU.

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
