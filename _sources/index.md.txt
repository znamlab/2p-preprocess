# 2p-preprocess

**Two-photon calcium imaging preprocessing pipeline** for the [Znamenskiy lab](https://github.com/znamlab).

---

`2p-preprocess` wraps [Suite2p](https://suite2p.readthedocs.io/) for ROI detection and fluorescence extraction, and adds a full post-processing pipeline including:

- **Bidirectional artefact correction** and motion estimation for z-stacks
- **Neuropil correction** via the Asymmetric Student's t-model (ASt) or a fixed coefficient
- **ΔF/F extraction** using Gaussian Mixture Model baseline estimation
- **Spike deconvolution** via the OASIS algorithm
- **Recording splitting** — splitting Suite2p's concatenated output back into per-recording chunks registered in Flexilims

```{toctree}
:maxdepth: 2
:caption: Getting started

installation
usage
pipeline
```

```{toctree}
:maxdepth: 1
:caption: Reference

autoapi/index
changelog
```

---

## Quick start

```bash
# 1. Clone and install
git clone git@github.com:znamlab/2p-preprocess.git
cd 2p-preprocess
pip install -e .

# 2. Run the full calcium pipeline for a session
2p calcium \
    --project my_project \
    --session MOUSE.1a_S20240101 \
    --conflicts overwrite \
    --tau 0.7
```

See the {doc}`installation` and {doc}`usage` pages for full details.
