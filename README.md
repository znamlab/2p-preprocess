# 2p-preprocess

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=readthedocs)](https://znamlab.github.io/2p-preprocess/)

## Documentation

The full documentation for this package, including installation guides, usage examples, and detailed pipeline descriptions, is available online:

👉 **[https://znamlab.github.io/2p-preprocess/](https://znamlab.github.io/2p-preprocess/)**

## Installation

We recommend you install `2p-preprocess` using [uv](https://docs.astral.sh/uv/).
uv is a modern Python package manager which is significantly faster than pip or conda, and has additional capabilities such as the generation of cross-platform lockfiles and per-project management and installation of python versions.

To install `2p-preprocess`:

Clone the repo from github:
```
git clone git@github.com:znamlab/2p-preprocess.git
```

This package includes [suite2p](https://suite2p.readthedocs.io/en/latest/) as an [optional dependency](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#dependencies-and-requirements).
This is required to run the full pipeline.
However, if you do not require the steps of the pipeline which call suite2p (e.g., because you are using this package as a library for another project), you can omit this extra.

To install the full pipeline, with suite2p:

```bash
ml uv

uv sync --extra suite2p

```

By default, this will install `2p-preprocess` into `.venv` in your current directory, using the dependencies specified in `uv.lock`.
If you wish to specify a different virtual environment (e.g., because your home partition is full), use the [`UV_PROJECT_ENVIRONMENT`](https://docs.astral.sh/uv/reference/environment/#uv_project_environment) environment variable.

The version of suite2p used by the pipeline (v0.14), requires **python < 3.10**.
Consequently, the default python version used by uv (specified in the `.python-version` file) is 3.9.
If you are not using the suite2p extra and wish to use a different python version, specify this when creating the virtual environment:

```bash

# We omit --extra suite2p
uv sync --python 3.11

```

## Using the pipeline

`run_suite2p.sh` and `run_suite2p_gpu.sh` contain example scripts that first runs the standard run_suite2p pipeline and then applies neuropil correction using the AST model.
If running neuropil correction using the AST model, using a GPU node is recommended.

To start the slurm job, navigate to the `2p-preprocess` directory.
Put the steps you want to run to y, and the steps you don’t want to run to n, e.g.:
```
--run-suite2p n --run-neuropil y --run-dff y
```
and run the`sbatch` script, passing the session details as environment variables, e.g.:
```bash
sbatch --export=PROJECT=depth_mismatch_seq,SESSION=BRAC9057.4j_S20240517,CONFLICTS=overwrite,TAU=0.7,UV_PROJECT_ENVIRONMENT=/nemo/lab/znamenskiyp/home/users/example-user/venvs/2p-preprocess run_suite2p_gpu.sh
```

There is a separate script for convenience if you want to run without AST neuropil:
```bash
sbatch --export=PROJECT=colasa_3d-vision_revisions,SESSION=PZAH17.1e_S20250311,CONFLICTS=overwrite,TAU=0.7,UV_PROJECT_ENVIRONMENT=/nemo/lab/znamenskiyp/home/users/example-user/venvs/2p-preprocess run_suite2p_gpu_noneuropil.sh
```

Note that if you do not export `UV_PROJECT_ENVIRONMENT`, the pipeline will fail, as the correct virtual environment will not be activated.

# ASt model
The Asymmetric Student's t-model for neuropil correction is described [here](https://basellasermouse.github.io/ast_model/model.html). The python implementation
in this repository uses [JAX](https://github.com/google/jax) for automatic
differentiation and rapid GPU computation. If run on a node without GPU, it
should revert to CPU.

# About
Some code in this repository (`extractdff_gmm`, `ast_model.py`) originates from a different code
base and is reused with permission of the original author, Maxime Rio.
