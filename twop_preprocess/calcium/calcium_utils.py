from functools import partial
from pathlib import Path
from numba import njit, prange
import numpy as np
import warnings
from sklearn import mixture
from tifffile import TiffFile
from tqdm import tqdm

print = partial(print, flush=True)


def get_weights(ops):
    """Get the weights for the suite2p detection.

    suite2p does not extract raw fluorescence traces, but rather uses a weighted sum of
    the raw fluorescence traces to detect ROIs. This function calculates the weights
    as in suite2p.detection.anatomical.select_rois (L161).
    For anatomical == 3 the weights are the intensity of each pixel relative to the
    1st and 99th percentile of the mean image + 0.1.

    Args:
        ops (dict): suite2p ops dictionary

    Returns:
        weights (numpy.ndarray): weights for the suite2p detection
    """
    mean_img = ops["meanImg"]
    if ops.get("denoise", 1):
        warnings.warn("Calculating weights on non-denoised data. F will change")

    if ops["anatomical_only"] == 1:
        raise NotImplementedError("anatomical_only=1 not implemented.")
        img = np.log(np.maximum(1e-3, max_proj / np.maximum(1e-3, mean_img)))
        weights = max_proj
    elif ops["anatomical_only"] == 2:
        # img = mean_img
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1))
            / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)),
            0,
            1,
        )
    elif ops["anatomical_only"] == 3:
        weights = 0.1 + np.clip(
            (mean_img - np.percentile(mean_img, 1))
            / (np.percentile(mean_img, 99) - np.percentile(mean_img, 1)),
            0,
            1,
        )
    else:
        raise NotImplementedError("Non anatomical not implemented. Requires max_proj")
        img = max_proj.copy()
        weights = max_proj
    return weights


@njit(parallel=True)
def rolling_percentile(arr, window, percentile):
    output = np.empty(len(arr) - window + 1)
    for i in prange(len(output)):
        output[i] = np.percentile(arr[i : i + window], percentile)
    return output


@njit(parallel=True, cache=True)  # cache=True for potential speedup on subsequent runs
def _calculate_rolling_baseline_parallel(F_segment, win_frames, percentile):
    """
    Calculates the rolling percentile baseline for a segment of fluorescence traces in parallel.

    Args:
        F_segment (np.ndarray): Fluorescence trace segment (n_rois, n_frames_in_segment).
        win_frames (int): Window size for rolling percentile.
        percentile (float): Percentile to calculate.

    Returns:
        np.ndarray: Rolling baseline for the segment (n_rois, n_frames_in_segment).
    """
    pad_before = win_frames // 2
    if win_frames % 2 == 0:
        pad_after = win_frames // 2 - 1
    else:
        pad_after = win_frames // 2
    n_rois, n_frames_seg = F_segment.shape
    # Pre-allocate output array with the same dtype as input
    rec_rolling_baseline = np.empty_like(F_segment)

    # Handle edge case where window is larger than segment length or <= 0
    # This check is now primarily done in the calling function, but a safeguard here is good.
    if win_frames <= 0:
        # Or raise an error, depending on desired behavior
        return np.zeros_like(F_segment)
    if win_frames > n_frames_seg:
        # If window is larger than segment, percentile of the whole segment is the best we can do
        for j in prange(n_rois):
            baseline_val = np.percentile(F_segment[j, :], percentile)
            rec_rolling_baseline[j, :] = baseline_val
        return rec_rolling_baseline
    if win_frames == 1:
        # Rolling percentile with window 1 is just the array itself
        return F_segment.copy()

    # Parallel loop over ROIs
    for j in prange(n_rois):
        # Calculate the core rolling percentile result
        # Ensure rolling_percentile handles edge cases like empty slices if needed,
        # though the win_frames > n_frames_seg check above should prevent basic issues.
        baseline_core = rolling_percentile(
            F_segment[j, :],  # Pass the 1D array for the current ROI
            win_frames,
            percentile,
        )

        # Manually apply edge padding (often more Numba-friendly than np.pad)
        # Ensure the target array for padding exists and has the right size
        padded_baseline = np.empty(n_frames_seg, dtype=baseline_core.dtype)

        # Fill the middle part with the calculated baseline
        # The core result has length n_frames_seg - win_frames + 1
        # It corresponds to frames from index (win_frames//2) up to (n_frames_seg - (win_frames//2)) approx.
        # The exact indices depend on the padding calculation.
        start_idx = pad_before
        end_idx = n_frames_seg - pad_after
        # Ensure indices match the length of baseline_core
        if len(baseline_core) == (end_idx - start_idx):
            padded_baseline[start_idx:end_idx] = baseline_core
        else:
            # This case indicates a potential mismatch in padding/window logic
            # Fallback or raise error - using edge padding as a simple fallback
            if len(baseline_core) > 0:
                padded_baseline[start_idx:end_idx] = baseline_core[
                    0
                ]  # Or some other strategy
            else:  # baseline_core is empty, maybe window > length? Handled above.
                padded_baseline[start_idx:end_idx] = 0  # Or np.nan

        # Apply edge padding
        if len(baseline_core) > 0:
            padded_baseline[:start_idx] = baseline_core[0]
            padded_baseline[end_idx:] = baseline_core[-1]
        else:
            # If baseline_core is empty (e.g., win_frames > n_frames_seg, though handled above),
            # pad with a default value, maybe the segment mean or 0.
            # This part might be redundant due to checks above.
            segment_median = np.median(F_segment[j, :]) if n_frames_seg > 0 else 0
            padded_baseline[:start_idx] = segment_median
            padded_baseline[end_idx:] = segment_median

        rec_rolling_baseline[j, :] = padded_baseline

    return rec_rolling_baseline


def correct_neuropil_ast(dpath, Fr, Fn):
    """
    Correct neuropil contamination using the ASt method.

    Args:
        dpath (str): path to the suite2p folder
        Fr (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois
            extracted from suite2p
        Fn (numpy.ndarray): shape nrois x time, neuropil fluorescence trace for all rois
            extracted from suite2p

    Returns:
        Fast (numpy.ndarray): shape nrois x time, neuropil corrected fluorescence trace
            for all rois extracted from suite2p

    """

    from twop_preprocess.neuropil.ast_model import ast_model

    stat = np.load(dpath / "stat.npy", allow_pickle=True)

    print("Starting neuropil correction with ASt method...", flush=True)
    traces, var_params, elbos = [], [], []
    for _Fr, _Fn, _stat in tqdm(zip(Fr, Fn, stat)):
        trace, param, elbo = ast_model(
            np.vstack([_Fr, _Fn]),
            np.array([_stat["npix"], _stat["neuropil_mask"].shape[0]]),
        )
        traces.append(trace)
        var_params.append(param)
        elbos.append(elbo)

    print("Neuropil correction completed... Saving...", flush=True)
    Fast = np.vstack(traces)
    np.save(dpath / "Fast.npy", Fast, allow_pickle=True)
    np.save(dpath / "ast_stat.npy", np.vstack(var_params), allow_pickle=True)
    np.save(dpath / "ast_elbo.npy", np.vstack(elbos), allow_pickle=True)
    return Fast


def correct_neuropil_standard(F, Fneu, neucoeff, save_path=None):
    """
    Applies standard neuropil correction: F = F - neucoeff * (Fneu - median(Fneu)).

    Args:
        F (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois
        Fneu (numpy.ndarray): shape nrois x time, neuropil fluorescence trace for all
            rois
        neucoeff (float): neuropil correction coefficient
        save_path (str, optional): path to save the neuropil corrected fluorescence
            trace. If None, will not save. Default None.

    Returns:
        F_corrected (numpy.ndarray): shape nrois x time, neuropil corrected fluorescence
            trace
    """

    Fneu_demeaned = Fneu - np.median(Fneu, axis=1, keepdims=True)
    F_corrected = F - neucoeff * Fneu_demeaned
    if save_path is not None:
        np.save(save_path, F_corrected, allow_pickle=True)
    return F_corrected


def dFF(f, n_components=2):
    """
    Helper function for calculating dF/F from raw fluorescence trace.
    Args:
        f (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois
            extracted from suite2p
        n_components (int): number of components for GMM. default 2.

    Returns:
        dffs (numpy.ndarray): shape nrois x time, dF/F for all rois extracted from
            suite2p

    """
    f0 = np.zeros(f.shape[0])
    for i in tqdm(range(f.shape[0])):
        gmm = mixture.GaussianMixture(n_components=n_components, random_state=42).fit(
            f[i].reshape(-1, 1)
        )
        gmm_means = np.sort(gmm.means_[:, 0])
        f0[i] = gmm_means[0]
    f0 = f0.reshape(-1, 1)
    dff = (f - f0) / f0
    return dff, f0


def calculate_and_save_dFF(dpath, F, filename_suffix, n_components=2):
    """
    Calculate dF/F for the whole session with concatenated recordings after neuropil
        correction.

    Args:
        dpath (str): path to the suite2p folder
        F (numpy.ndarray): shape nrois x time, neuropil
        filename_suffix (str): suffix to add to the filename
        n_components (int): number of components for GMM. default 2.

    Returns:
        dff (numpy.ndarray): shape nrois x time, dF/F for all rois extracted from
            suite2p
        f0 (numpy.ndarray): shape nrois, f0 for each roi
    """
    print("Calculating dF/F...")
    # Calculate dFFs and save to the suite2p folder
    print(f"n components for dFF calculation: n_components")
    dff, f0 = dFF(F, n_components=n_components)
    np.save(dpath / f"dff{filename_suffix}.npy", dff)
    np.save(dpath / f"f0{filename_suffix}.npy", f0)
    return dff, f0


def estimate_offset(datapath, n_components=3):
    """
    Estimate the offset for a given tiff file using a GMM with n_components.


    Args:
        datapath (str): path to the tiff file
        n_components (int): number of components for GMM. default 3.

    Returns:
        offset (float): estimated offset

    """
    # find the first tiff at the path
    tiffs = list(Path(datapath).glob("*.tif"))
    if len(tiffs) == 0:
        raise ValueError(f"No tiffs found at {datapath}")
    tiff = tiffs[0]
    # load the tiff using tifffile
    with TiffFile(tiff) as tif:
        # get the first frame
        frame = tif.asarray(key=0)
    # find the offset
    gmm = mixture.GaussianMixture(n_components=n_components, random_state=42).fit(
        frame.reshape(-1, 1)
    )
    gmm_means = np.sort(gmm.means_[:, 0])
    return gmm_means[0]


def correct_offset(datapath, offsets, first_frames, last_frames):
    """
    Load the concatenated fluorescence trace and subtract offset for each recording.

    Args:
        datapath (str): path to the concatenated fluorescence trace
        offsets (numpy.ndarray): shape nrecordings, offsets for each recording
        first_frames (numpy.ndarray): shape nrecordings, first frame of each recording
        last_frames (numpy.ndarray): shape nrecordings, last frame of each recording

    Returns:
        F (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois extracted from suite2p

    """
    # load the concatenated fluorescence trace
    F = np.load(datapath)
    # subtract offset for each recording
    for start, end, offset in zip(first_frames, last_frames, offsets):
        F[:, start:end] -= offset
    return F


def get_recording_frames(suite2p_dataset):
    """
    Get the first and last frames of each recording in the session.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings


    Returns:
        first_frames (numpy.ndarray): shape nrecordings x nplanes, first frame of each recording
        last_frames (numpy.ndarray): shape nrecordings x nplanes, last frame of each recording

    """
    # load the ops file to find length of individual recordings
    try:
        nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))

    except KeyError:  # Default to 1 if missing
        suite2p_dataset.extra_attributes["nplanes"] = 1
        suite2p_dataset.update_flexilims(mode="update")

        nplanes = int(suite2p_dataset.extra_attributes["nplanes"])

    ops = []
    for iplane in range(nplanes):
        ops_path = suite2p_dataset.path_full / f"plane{iplane}" / "ops.npy"
        ops.append(np.load(ops_path, allow_pickle=True).item())
    # different planes may have different number of frames if recording is stopped mid-volume
    last_frames = []
    first_frames = []
    for ops_plane in ops:
        last_frames.append(np.cumsum(ops_plane["frames_per_folder"]))
        first_frames.append(np.concatenate(([0], last_frames[-1][:-1])))
    last_frames = np.stack(last_frames, axis=1)
    first_frames = np.stack(first_frames, axis=1)
    return first_frames, last_frames
