# Changelog

All notable changes to `2p-preprocess` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- **ROI Pipeline Visualization**: Implemented a unified single-ROI diagnostic plot (`roi_pipelines/`) that shows the full processing evolution (Raw -> Offset -> Detrend -> Processed -> ΔF/F) for individual cells. Includes recording boundaries and estimated offsets for better jump/drift identification.
- **Population Health Diagnostics**: Added a population-level quality metric plot (`05b_population_metrics.png`) showing distributions of baseline fluorescence (F0), median ΔF/F, and extreme transients (>10000%).
- **Sanity CLI Command**: Added a dedicated `2p sanity` command to re-generate diagnostic plots without re-running the entire processing pipeline. Supports an `--annotated` flag to handle sessions processed via `2p reextract`.
- **Targeted Quality Control**: Refactored GMM baseline fit plotting (`gmm_offsets/`) to prioritize "problematic" ROIs (e.g., negative F0, negative median ΔF/F, or extreme transients) for faster manual validation of session health.
- **Performance Feedback**: Added `tqdm` progress bars and detailed console logging to the detrending stage to provide real-time status updates during long-running batch jobs.
- **Technical Pipeline Documentation**: Expanded `docs/pipeline.md` with a detailed technical overview of the `2p calcium` command, documenting the exact calculation scopes (per-recording vs. per-neuron vs. per-session) for optical offsets, rolling baseline detrending, and GMM-based ΔF/F.
- **Optical Offset Diagnostics**: New sanity plot for optical offset estimation. It displays the raw pixel intensity histogram of the first frame alongside the fitted GMM components and the selected offset value.
- Memory-safe subsampling in `estimate_offset` to handle high-resolution raw TIFFs without excessive memory usage.

### Changed

- **Optical Offset Diagnostics**: Expanded the sanity plot for optical offset estimation with dual linear/log scales for better visualization of low-intensity pixel distributions.
- **Flexiznam Integration**: Standardized internal API calls to use the `project` parameter consistently across data retrievals and dataset creation.
- **Refactored ROI Metrics**: Modularized ROI quality metrics and diagnostic selection logic to ensure consistency between console reporting and visual diagnostic plots.
- **Renamed Fitting Functions**: Renamed GMM baseline fitting functions and their corresponding output folders (e.g., `gmm_offsets/`) for better technical accuracy and clarity.

### Fixed

- **Pipeline Robustness**: Hardened the sanity check pipeline to handle missing intermediate files (`Fast.npy`, `Fstandard.npy`) gracefully by generating placeholder warning plots instead of crashing.
- **Visualization Accuracy**: Updated `plot_offset_gmm` to use the session-specific `neucoeff`, ensuring the neuropil subtraction visualization accurately reflects the processing parameters.
- **Matplotlib Compatibility**: Resolved `DeprecationWarning` regarding array-to-scalar conversion in plotting routines.
- **Numba Optimization**: Refactored `rolling_percentile` to remove an invalid `parallel=True` directive, resolving performance warnings and ensuring stable execution.
- **Visualization Robustness**: Added console warnings when non-finite values (NaN/Inf) are detected and filtered during the plotting of neural traces or GMM fits.

---

## [0.1.0] — 2026-04-23

First tagged release. The pipeline was already functional; this release adds
proper packaging, documentation, and CI infrastructure.

### Added

- **Documentation site** (`docs/`) built with Sphinx, MyST-Parser, furo theme,
  and `sphinx-autoapi` (auto-generates API reference without importing the package,
  so heavy/private dependencies are not required at build time).
  - `docs/installation.md` — full install guide covering standard, Suite2p-compatible,
    and GPU-accelerated installs.
  - `docs/usage.md` — CLI reference for `2p calcium`, `2p reextract`, and `2p zstack`,
    plus SLURM batch scripts and Python API examples.
  - `docs/pipeline.md` — step-by-step pipeline narrative (Suite2p extraction,
    offset correction, detrending, neuropil correction, ΔF/F, spike deconvolution,
    recording splitting, z-stack registration) with parameter tables.
  - `docs/changelog.md` — this file.
- **GitHub Actions workflow** (`.github/workflows/docs.yml`) — builds docs on every
  pull request and deploys automatically to GitHub Pages on every push to `main`.
- **Git-tag-based versioning** via `setuptools_scm`:
  - `version_file = "twop_preprocess/_version.py"` added to `[tool.setuptools_scm]`
    so the version is written at install time and accessible at runtime as
    `twop_preprocess.__version__`.
  - `docs/conf.py` reads the version from git at build time via
    `setuptools_scm.get_version()`.
- `docs` optional-dependency group in `pyproject.toml`
  (`pip install -e ".[docs]"` installs Sphinx and friends).
- Docs badge in `README.md`.

### Fixed

- `twop_preprocess/plotting_utils/s2p_movie_utils.py` — corrected a Python
  `SyntaxError` in `write_moving_average_tif`: positional arguments `out_dir` and
  `fname` appeared after the default argument `w=100`, which is invalid syntax.
  Argument order is now `(im, out_dir, fname, w=100)`.

### Changed

- `twop_preprocess/__init__.py` — now exposes `__version__`.
- `.gitignore` — added `twop_preprocess/_version.py` (auto-generated, not for VCS).
- **Docstring Standardization**: Completed comprehensive Google-style docstring coverage across all core modules (`calcium.py`, `calcium_s2p.py`, `calcium_utils.py`, `neuropil.py`, `ast_model.py`, `processing_steps.py`, `utils.py`, and `plotting_utils`). This ensures high-quality auto-generated API documentation.
- `README.md` — added dedicated Documentation section with a link to the hosted GitHub Pages site.

[Unreleased]: https://github.com/znamlab/2p-preprocess/compare/v0.1.0...dev
 [0.1.0]: https://github.com/znamlab/2p-preprocess/releases/tag/v0.1.0
