# Installation

## Prerequisites

- Python ≥ 3.8 (< 3.10 if using Suite2p, see below)
- A configured [Flexilims](https://github.com/znamlab/flexiznam) environment (the pipeline stores processed datasets in Flexilims)
- SSH access to the `znamlab` private GitHub organisation (required to install `flexiznam` and `znamutils`)

---

## Standard install (no Suite2p)

This is the recommended install for users who will **submit SLURM jobs** that call Suite2p on a cluster node, rather than running Suite2p locally.

```bash
git clone git@github.com:znamlab/2p-preprocess.git
cd 2p-preprocess
pip install -e .
```

This installs all pipeline dependencies (`jax`, `optax`, `tifffile`, `scikit-image`, etc.) but excludes Suite2p itself.

---

## Suite2p-compatible install (Python < 3.10)

Suite2p currently requires Python < 3.10. Use the provided conda environment file, which pins the correct versions:

```bash
git clone git@github.com:znamlab/2p-preprocess.git
cd 2p-preprocess

# Create and activate the environment
conda env create -f environment.yml
conda activate 2p-preprocess

# Install the package in editable mode
pip install -e .

# Install JAX with CUDA 12 support (for GPU acceleration)
pip install -U "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Pin optax to a version compatible with Python 3.8
pip install optax==0.1.2

conda deactivate
```

> **Note:** `torch` and `optax` version conflicts arise with Python 3.8. The environment file handles this, but manual installs may require adjustments.

---

## Installing docs dependencies (developers)

```bash
pip install -e ".[docs]"
```

Then build the docs locally:

```bash
sphinx-build -b html docs docs/_build/html
# Open docs/_build/html/index.html in your browser
```

---

## User configuration file

The pipeline reads an optional YAML configuration file from `~/.2p_preprocess/config.yml`.
Any key present here overrides the built-in defaults (see {doc}`pipeline` for a full list of options).

Example:

```yaml
# ~/.2p_preprocess/config.yml
tau: 0.7
ast_neuropil: true
neucoeff: 0.7
```
