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

Removes the systematic optical offset (dark current) that arises from the PMT and electronics. This offset must be removed before any multiplicative corrections (like detrending or ΔF/F) are applied.

*   **Calculation**: For each individual recording in the session, the pipeline loads the first frame of the raw ScanImage TIFF. It fits a 3-component Gaussian Mixture Model (GMM) to the pixel intensity distribution of this frame. The mean of the lowest Gaussian component is taken as the optical offset.
*   **Scope**: Calculated **per recording**.
*   **Application**: The offset is subtracted from the concatenated fluorescence traces on a per-recording basis.
*   **Implementation**:
     *   [`estimate_offset`](https://github.com/znamlab/2p-preprocess/blob/dev/twop_preprocess/calcium/calcium_utils.py#L318) (in `calcium_utils.py`) is the core function that calculates the offset for a **single recording** by fitting the GMM to its first frame.
     *   [`estimate_offsets`](https://github.com/znamlab/2p-preprocess/blob/dev/twop_preprocess/calcium/processing_steps.py#L71) (in `processing_steps.py`) is the wrapper that orchestrates this across all recordings in the session.

### 2b — Detrending (`detrend`)

Removes slow baseline drifts caused by photobleaching or changes in focus.

*   **Calculation**: A rolling percentile filter (default 20th percentile) is applied to each ROI's fluorescence trace.
*   **Scope**: Calculated **per recording** and **per neuron**.
*   **Normalization**: To avoid baseline "jumps" between concatenated recordings, the baseline for each recording segment is aligned to the median baseline of the **first recording** in the session.
*   **Method**:
    *   `subtract`: `F_corrected = F - (baseline - first_rec_baseline_median)`
    *   `divide`: `F_corrected = F / (baseline / first_rec_baseline_median)`
*   **Implementation**: [`detrend`](https://github.com/znamlab/2p-preprocess/blob/dev/twop_preprocess/calcium/processing_steps.py#L11) in `twop_preprocess/calcium/processing_steps.py`.

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

*   **Implementation**: [`ast_model`](https://github.com/znamlab/2p-preprocess/blob/dev/twop_preprocess/neuropil/ast_model.py#L51) in `twop_preprocess/neuropil/ast_model.py`.

**Option B — Fixed coefficient** (`ast_neuropil: False`):

```
F_corrected = F - neucoeff * (Fneu - median(Fneu))
```

| Parameter | Default | Description |
|---|---|---|
| `ast_neuropil` | `True` | Use ASt model |
| `neucoeff` | `0.7` | Fixed neuropil coefficient (Option B only) |

### 2d — ΔF/F extraction

Calculates ΔF/F using a Gaussian Mixture Model (GMM) to estimate the baseline fluorescence $F_0$.

*   **Calculation**: For each neuron, a GMM with `dff_ncomponents` (default 2) is fitted to the distribution of all fluorescence values across the **entire session**. The mean of the component with the lowest mean value is defined as $F_0$.
*   **Scope**: Calculated **per session** and **per neuron**.
*   **Formula**: $\Delta F/F = (F - F_0) / F_0$
*   **Implementation**: [`dFF`](https://github.com/znamlab/2p-preprocess/blob/dev/twop_preprocess/calcium/calcium_utils.py#L261) in `twop_preprocess/calcium/calcium_utils.py`.

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

 When `sanity_plots: True` (default), the pipeline generates diagnostic figures in a `sanity_plots/` subdirectory within each plane folder and at the top level of the dataset. These include:

 *   **Optical Offset Diagnostics**: Shows the GMM fit to the raw pixel intensity histogram for each recording, illustrating how the optical offset was estimated.
 *   **Trace Diagnostics**: Raw traces, offset-corrected traces, detrended traces, and neuropil-corrected traces.
 *   **ΔF/F Diagnostics**: ΔF/F traces and the GMM fit used for $F_0$ baseline estimation.
 *   **Fluorescence Matrices**: Normalized heatmaps of `F`, `Fneu`, and corrected traces across all neurons.

 ### Re-running diagnostics

 If you want to re-generate the sanity plots without re-running the entire pipeline (e.g., after updating the plotting code or to change the number of ROIs plotted), you can use the `sanity` command:

 ```bash
 twop_preprocess sanity --project <project_name> --session <session_name>
 ```

 This command loads the existing processed data, re-calculates the necessary intermediate steps (like detrending), and overwrites the files in the `sanity_plots/` directories.
