# Changelog

All notable changes to `2p-preprocess` will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

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
