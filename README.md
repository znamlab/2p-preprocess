# Installing the pipeline

Clone the repo from github:
```
git clone git@github.com:znamlab/2p-preprocess.git
```

Next, navigate to the repo directory and run following commands to install the package
and dependencies:
```
cd 2p-preprocess
conda env create -f environment.yml

conda activate 2p-preprocess
pip install -e .
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax
conda deactivate
```

Until we sort out whether we can use the newest jax in the cluster (whoever does it gets a coffee) as that's what limits us from updating the versions, there is an environment with the explicit working versions for all the dependencies in June 2024 as `environment.yaml`. Use if there are version conflicts.  

This should install the dependencies and create conda environments for suite2p
and for the repo itself. Environments are created in each users home directory.

`run_suite2p.sh` and `run_suite2p_gpu.sh` contain example scripts that first runs the standard run_suite2p pipeline and then applies neuropil correction using the AST model.
If running neuropil correction using the AST model, using a GPU node is recommended.

To start the slurm job, navigate to the `2p-preprocess` directory.
Put the steps you want to run to y, and the steps you don’t want to run to n, e.g.:
```
--run-suite2p n --run-neuropil y --run-dff y
```
and run the`sbatch` script, passing the session details as environment variables, e.g.:
```
sbatch --export=PROJECT=test,SESSION=PZAJ2.1c_S20210513,CONFLICTS=skip,TAU=0.7 run_suite2p_gpu.sh
```

# ASt model
The Asymmetric Student's t-model for neuropil correction is described [here](https://basellasermouse.github.io/ast_model/model.html). The python implementation
in this repository uses [JAX](https://github.com/google/jax) for automatic
differentiation and rapid GPU computation. If run on a node without GPU, it
should revert to CPU.

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
