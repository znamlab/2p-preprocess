# Pipeline overview

This page describes the full preprocessing pipeline in the order that steps are executed. All steps are controlled by the `ops` dictionary (see {doc}`usage` for how to override defaults from the CLI or Python).

---

## Step 0 — Prerequisites: Flexilims bookkeeping

The pipeline is designed to work with a [Flexilims](https://github.com/znamlab/flexiznam) data management system. Raw ScanImage TIFF recordings must already be registered as `scanimage` datasets under the relevant session in Flexilims before running the pipeline.

---

## Step 1 — Suite2p extraction (`run_suite2p`)

Suite2p is used for:

1. **Motion correction** — aligning each frame to a reference image using phase correlation.
2. **ROI detection** — cell segmentation using either Cellpose (anatomical, `roidetect=True` with `anatomical_only=3`) or the default Suite2p algorithm.
3. **Fluorescence extraction** — extracting mean fluorescence `F` (ROI) and `Fneu` (neuropil ring) for each ROI across all frames.

Because Suite2p concatenates all recordings in a session into a single timeline, the raw outputs are stored as a single `suite2p_rois` dataset in Flexilims.

**Key ops:**

| Parameter | Default | Description |
|---|---|---|
| `roidetect` | `1` | Run ROI detection |
| `anatomical_only` | `3` | Use Cellpose segmentation |
| `pretrained_model` | `cyto2` | Cellpose model |
| `diameter_multiplier` | `0.01` | Scale factor for Cellpose diameter |
| `threshold_scaling` | `0.5` | Suite2p detection threshold scale |
| `delete_bin` | `True` | Delete registered binary after extraction |

---

## Step 2 — Post-processing per plane (`run_dff`)

After Suite2p extraction, the following per-plane corrections are applied to the **concatenated** fluorescence traces. Each step can be toggled via `ops`.

### 2a — Offset correction (`correct_offset`)

Removes a systematic fluorescence offset that arises between different recordings within the session (e.g., due to PMT gain changes between runs). The offset is estimated as the median baseline fluorescence in a dark period at the start of each recording.

### 2b — Detrending (`detrend`)

Removes slow drift in fluorescence using a rolling percentile filter. This corrects for photobleaching and long-term baseline shifts.

| Parameter | Default | Description |
|---|---|---|
| `detrend` | `True` | Enable detrending |
| `detrend_win` | `60.0` | Window length (seconds) |
| `detrend_pctl` | `20.0` | Percentile for baseline estimate |
| `detrend_method` | `subtract` | `subtract` or `divide` |

### 2c — Neuropil correction

Removes contamination of the ROI signal by the surrounding neuropil.

**Option A — ASt model** (`ast_neuropil: True`, recommended):

The Asymmetric Student's t (ASt) model fits a probabilistic model jointly to `F` and `Fneu` for each ROI, estimating the neuropil contamination coefficient and the true neuronal signal simultaneously using variational inference with JAX. See the [ASt model documentation](https://basellasermouse.github.io/ast_model/model.html) for the statistical derivation.

Outputs: `Fast.npy` (corrected fluorescence), `ast_stat.npy` (variational parameters), `ast_elbo.npy` (ELBO).

**Option B — Fixed coefficient** (`ast_neuropil: False`):

```
F_corrected = F - neucoeff * Fneu
```

| Parameter | Default | Description |
|---|---|---|
| `ast_neuropil` | `True` | Use ASt model |
| `neucoeff` | `0.7` | Fixed neuropil coefficient (Option B only) |

### 2d — ΔF/F extraction

Calculates ΔF/F using a Gaussian Mixture Model (GMM) to estimate the baseline fluorescence F₀:

```
dF/F = (F - F₀) / F₀
```

The GMM decomposes the fluorescence distribution into `dff_ncomponents` Gaussian components; the component with the lowest mean is taken as the baseline.

| Parameter | Default | Description |
|---|---|---|
| `dff_ncomponents` | `2` | Number of GMM components |

### 2e — Spike deconvolution

Suite2p's OASIS algorithm is used to deconvolve inferred spike rates from the ΔF/F trace. The `tau` parameter controls the assumed calcium indicator decay time constant.

| Parameter | Default | Description |
|---|---|---|
| `tau` | `0.7` | Decay time constant (seconds). Use ~0.7 for GCaMP6s, ~0.4 for GCaMP8m |
| `baseline_method` | `maximin` | Baseline estimation method for deconvolution |
| `sig_baseline` | `10.0` | Smoothing sigma for baseline |
| `win_baseline` | `60.0` | Window for baseline estimation (seconds) |

---

## Step 3 — Recording splitting (`run_split`)

Suite2p concatenates all recordings within a session. After processing, the pipeline cuts the concatenated traces back into per-recording segments and registers each as a `suite2p_traces` dataset in Flexilims (as a child of the corresponding `recording` entity). This makes it straightforward to load traces aligned to individual experimental recordings in downstream analyses.

---

## Z-stack registration

The `2p zstack` command (see {doc}`usage`) runs a separate motion-correction pipeline for anatomical z-stacks:

1. **Bidirectional scanning correction** — estimates and corrects the horizontal line offset between odd and even scan lines caused by bidirectional resonance scanning.
2. **Within-plane motion correction** — uses phase correlation to align individual frames within each z-plane and averages them.
3. **Between-plane alignment** — sequentially aligns adjacent z-planes to produce a coherent 3-D stack.

Registered stacks are saved as TIFF files and registered as `registered_stack` datasets in Flexilims.

---

## Output files (per plane)

After a full run, each `plane{N}` subdirectory of the `suite2p_rois` dataset contains:

| File | Description |
|---|---|
| `F.npy` | Raw ROI fluorescence (n_rois × n_frames) |
| `Fneu.npy` | Raw neuropil fluorescence |
| `spks.npy` | Suite2p deconvolved spikes |
| `Fast.npy` | ASt-corrected fluorescence *(if ASt enabled)* |
| `dff_ast.npy` | ΔF/F from ASt-corrected trace *(if ASt enabled)* |
| `spks_ast.npy` | Deconvolved spikes from ASt trace *(if ASt enabled)* |
| `dff.npy` | ΔF/F from standard neuropil-corrected trace *(if ASt disabled)* |
| `Fstandard.npy` | Standard neuropil-corrected fluorescence *(if ASt disabled)* |
| `stat.npy` | ROI statistics (Suite2p format) |
| `ops.npy` | Suite2p ops used for this run |
| `iscell.npy` | Cell classification array |

---

## Sanity plots

When `sanity_plots: True` (default), the pipeline generates diagnostic figures in a `sanity_plots/` subdirectory within each plane folder. These include raw traces, offset-corrected traces, detrended traces, neuropil-corrected traces, ΔF/F traces, and GMM fit diagnostics.
