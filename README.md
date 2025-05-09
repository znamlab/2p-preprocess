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
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax==0.1.2
conda deactivate
```

This should install the dependencies and create conda environments for `suite2p` and for the repo itself. Environments are created in each users home directory. Note that the version of `torch` generated issues and the version of `optax` conflicted with `python=3.8`. The python version is enforced because `suite2p` does not want to bring their dependencies forward.

`run_suite2p.sh` and `run_suite2p_gpu.sh` contain example scripts that first runs the standard run_suite2p pipeline and then applies neuropil correction using the AST model.
If running neuropil correction using the AST model, using a GPU node is recommended.

To start the slurm job, navigate to the `2p-preprocess` directory.
Put the steps you want to run to y, and the steps you don’t want to run to n, e.g.:
```
--run-suite2p n --run-neuropil y --run-dff y
```
and run the`sbatch` script, passing the session details as environment variables, e.g.:
```
sbatch --export=PROJECT=depth_mismatch_seq,SESSION=BRAC9057.4j_S20240517,CONFLICTS=overwrite,TAU=0.7 run_suite2p_gpu.sh
```

There is a separate script for convenience if you want to run without AST neuropil:
```
sbatch --export=PROJECT=colasa_3d-vision_revisions,SESSION=PZAH17.1e_S20250311,CONFLICTS=overwrite,TAU=0.7 run_suite2p_gpu_noneuropil.sh
```


# ASt model
The Asymmetric Student's t-model for neuropil correction is described [here](https://basellasermouse.github.io/ast_model/model.html). The python implementation
in this repository uses [JAX](https://github.com/google/jax) for automatic
differentiation and rapid GPU computation. If run on a node without GPU, it
should revert to CPU.

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
