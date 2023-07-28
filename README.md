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
conda install pip
pip install -e .
pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install optax
conda deactivate
```

Current fix for package version conflict:
- downgrade suite2p to ver 0.12.1 `pip install suite2p==0.12.1`
- downgrade numpy to ver 1.21.1  `pip install numpy==1.21.1`
- install jax-cuda11-cudnn8.2 `pip install jaxlib==0.4.7+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`
- downgrade jax to ver 0.4.7 or 0.4.8 `pip install jax==0.4.7`
- uninstall nvidia-cublas `pip uninstall nvidia-cublas-cu11`

This should install the dependencies and create conda environments for suite2p
and for the repo itself. Environments are created in each users home directory.

`run_suite2p.sh` and `run_suite2p_gpu.sh` contain example scripts that first runs the standard run_suite2p pipeline and then applies neuropil correction using the AST model.
If running neuropil correction using the AST model, using a GPU node is recommended.

To start the slurm job, navigate to the `2p-preprocess` directory and run the
`sbatch` script, passing the session details as environment variables, e.g.:
```
sbatch --export=PROJECT=test,SESSION=PZAJ2.1c_S20210513 run_suite2p_gpu.sh
```

# ASt model
The Asymmetric Student's t-model for neuropil correction is described [here](https://basellasermouse.github.io/ast_model/model.html). The python implementation
in this repository uses [JAX](https://github.com/google/jax) for automatic
differentiation and rapid GPU computation. If run on a node without GPU, it
should revert to CPU.

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
