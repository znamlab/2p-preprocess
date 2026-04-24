# Using the pipeline

The pipeline is controlled via the `2p` command-line interface (CLI) and/or directly through the Python API.

---

## Command-line interface (CLI)

After installing the package, the `2p` command is available in your environment.

### `2p calcium` — run the full calcium preprocessing pipeline

```
Usage: 2p calcium [OPTIONS]

  Run calcium imaging preprocessing pipeline

Options:
  -p, --project TEXT              Name of the project (Flexilims project name)
  -s, --session TEXT              Flexilims name of the session
  -c, --conflicts TEXT            How to handle conflicts when processed data
                                  already exists  [overwrite | skip | abort | append]
  --run-neuropil / --no-run-neuropil
                                  Whether to run ASt neuropil correction
  --run-split / --no-run-split    Whether to split concatenated recordings back
                                  into individual recordings  [default: True]
  --run-suite2p / --no-run-suite2p
                                  Whether to run Suite2p extraction  [default: True]
  --run-dff / --no-run-dff        Whether to run dF/F extraction  [default: True]
  -t, --tau FLOAT                 Decay time constant for spike extraction (seconds)
  --keep-binary                   Keep the Suite2p binary file after processing
  --roidetect BOOLEAN             Whether to run ROI detection on Suite2p output
  --help                          Show this message and exit.
```

**Example — full pipeline with ASt neuropil correction:**

```bash
2p calcium \
    --project depth_mismatch_seq \
    --session BRAC9057.4j_S20240517 \
    --conflicts overwrite \
    --run-neuropil \
    --tau 0.7
```

**Example — skip Suite2p, only run dF/F:**

```bash
2p calcium \
    --project depth_mismatch_seq \
    --session BRAC9057.4j_S20240517 \
    --no-run-suite2p \
    --no-run-split \
    --run-dff
```

---

### `2p reextract` — re-extract ROI masks

Re-extract fluorescence traces using a custom set of ROI masks (e.g., from a different segmentation run).

```
Usage: 2p reextract [OPTIONS]

  Re-extract masks for a session.

Options:
  -p, --project TEXT       Name of the project  [required]
  -s, --session TEXT       Flexilims name of the session  [required]
  -m, --masks-path PATH    Path to the .npy file with masks to re-extract  [required]
  -c, --conflicts TEXT     How to handle existing re-extracted data
                           [abort | skip | append | overwrite]  [default: abort]
  --use-slurm / --no-use-slurm
                           Submit as a SLURM job  [default: True]
  --help                   Show this message and exit.
```

**Example:**

```bash
2p reextract \
    --project my_project \
    --session MOUSE.1a_S20240101 \
    --masks-path /path/to/custom_masks.npy \
    --conflicts overwrite
```

---

### `2p zstack` — register a z-stack

Motion-correct and align planes of a two-photon z-stack.

```
Usage: 2p zstack [OPTIONS] [DATASETS]...

  Run zstack registration

Options:
  -p, --project TEXT          Name of the project
  -s, --session TEXT          Flexilims name of the session
  --conflicts TEXT            How to handle existing registered z-stacks
  -c, --channel INTEGER       Channel to use for registration  [default: 0]
  --max-shift INTEGER         Maximum pixel shift for registration
  --align-planes BOOLEAN      Whether to align planes to each other
  --iter INTEGER              Number of registration iterations
  --bidi-correction BOOLEAN   Apply bidirectional scanning artefact correction
  --sequential-volumes BOOLEAN
                              Stack was imaged as a sequence of volumes rather
                              than plane-by-plane
  --zstack-concat BOOLEAN     Concatenate multiple z-stack datasets  [default: False]
  --help                      Show this message and exit.
```

**Example:**

```bash
2p zstack \
    --project my_project \
    --session MOUSE.1a_S20240101 \
    --conflicts append \
    --bidi-correction True
```

---

### `2p sanity` — re-generate diagnostic plots

Re-generate all sanity plots for a previously processed session without re-running the entire pipeline.

```
Usage: 2p sanity [OPTIONS]

  Re-generate sanity plots for a session.

Options:
  -p, --project TEXT  Name of the project  [required]
  -s, --session TEXT  Flexilims name of the session  [required]
  --help              Show this message and exit.
```

**Example:**

```bash
2p sanity --project depth_mismatch_seq --session BRAC9057.4j_S20240517
```

---

## Running via SLURM (recommended for HPC)

The repository ships with example SLURM batch scripts. Navigate to the repo root and submit with `sbatch`, passing session details as environment variables:

**Full pipeline with ASt neuropil (GPU node):**

```bash
sbatch \
    --export=PROJECT=depth_mismatch_seq,SESSION=BRAC9057.4j_S20240517,CONFLICTS=overwrite,TAU=0.7 \
    run_suite2p_gpu.sh
```

**Full pipeline without ASt neuropil:**

```bash
sbatch \
    --export=PROJECT=colasa_3d-vision_revisions,SESSION=PZAH17.1e_S20250311,CONFLICTS=overwrite,TAU=0.7 \
    run_suite2p_gpu_noneuropil.sh
```

**Z-stack registration:**

```bash
sbatch \
    --export=PROJECT=my_project,SESSION=MOUSE.1a_S20240101 \
    register_zstack.sh
```

---

## Python API

You can also call the pipeline programmatically:

```python
from twop_preprocess.calcium import extract_session

extract_session(
    project="depth_mismatch_seq",
    session_name="BRAC9057.4j_S20240517",
    conflicts="overwrite",
    run_suite2p=True,
    run_dff=True,
    run_split=True,
    ops={"tau": 0.7, "ast_neuropil": True},
)
```

```python
from twop_preprocess.zstack import run_zstack_registration

run_zstack_registration(
    project="my_project",
    session_name="MOUSE.1a_S20240101",
    conflicts="append",
    ops={"bidi_correction": True, "max_shift": 50},
)
```

See the {doc}`autoapi/index` for full API documentation.
