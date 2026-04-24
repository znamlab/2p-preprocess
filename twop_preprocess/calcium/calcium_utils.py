from functools import partial
from pathlib import Path
from numba import njit, prange
import numpy as np
import warnings
from sklearn import mixture
from tifffile import TiffFile
from tqdm import tqdm
from ..plotting_utils import sanity_check_utils as sanity

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
    try:
        from suite2p import io
        from suite2p.detection.detect import bin_movie
        from suite2p.detection.denoise import pca_denoise
        from suite2p.detection.utils import temporal_high_pass_filter
    except ImportError:
        raise ImportError(
            "suite2p is not installed. Please install suite2p to use this function."
        )

    n_frames, Ly, Lx = ops["nframes"], ops["Ly"], ops["Lx"]
    filename = Path(ops["reg_file"]).resolve()
    if not filename.exists():
        raise FileNotFoundError(f"Binary file not found at: {filename}")
    if filename.stat().st_size == 0:
        raise ValueError(f"Binary file is empty (0 bytes): {filename}")

    with io.BinaryFile(
        Ly=Ly, Lx=Lx, filename=str(filename), n_frames=n_frames
    ) as f_reg:
        yrange = ops.get("yrange", [0, Ly])
        xrange = ops.get("xrange", [0, Lx])

        bin_size = int(
            max(1, n_frames // ops["nbinned"], np.round(ops["tau"] * ops["fs"]))
        )
        print("Binning movie in chunks of length %2.2d" % bin_size)
        mov = bin_movie(
            f_reg,
            bin_size,
            yrange=yrange,
            xrange=xrange,
            badframes=ops.get("badframes", None),
        )
        if ops.get("inverted_activity", False):
            mov -= mov.min()
            mov *= -1
            mov -= mov.min()

        if ops.get("denoise", 1):
            mov = pca_denoise(
                mov,
                block_size=[ops["block_size"][0] // 2, ops["block_size"][1] // 2],
                n_comps_frac=0.5,
            )
    mean_img = mov.mean(axis=0)
    mov = temporal_high_pass_filter(mov=mov, width=int(ops["high_pass"]))
    # max_proj = np.percentile(mov, 90, axis=0) #.mean(axis=0)
    if ops["anatomical_only"] == 1:
        weights = mov.max(axis=0)
    elif ops["anatomical_only"] == 2:
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
        weights = mov.max(axis=0)

    return weights


@njit(cache=True)
def rolling_percentile(arr, window, percentile):
    """
    Compute a rolling percentile over a 1D array.

    Args:
        arr (numpy.ndarray): 1D input array.
        window (int): Window size in samples.
        percentile (float): Percentile to compute (0-100).

    Returns:
        numpy.ndarray: The rolling percentile values (length = len(arr) - window + 1).
    """
    output = np.empty(len(arr) - window + 1)
    for i in range(len(output)):
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
    Correct neuropil contamination using the Asymmetric Student's t-model (ASt).

    Args:
        dpath (Path): Path to the Suite2p plane folder.
        Fr (numpy.ndarray): Raw fluorescence traces (n_rois x n_frames).
        Fn (numpy.ndarray): Neuropil fluorescence traces (n_rois x n_frames).

    Returns:
        np.ndarray: The ASt-corrected fluorescence traces (Fast).
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
    Apply standard neuropil correction to fluorescence traces.

    Formula: F_corrected = F - neucoeff * (Fneu - median(Fneu)).

    Args:
        F (numpy.ndarray): Raw fluorescence traces (n_rois x n_frames).
        Fneu (numpy.ndarray): Neuropil fluorescence traces (n_rois x n_frames).
        neucoeff (float): Neuropil correction coefficient (usually 0.7).
        save_path (Path or str, optional): Path to save the corrected trace as a
            .npy file. If None, does not save. Default None.

    Returns:
        np.ndarray: The neuropil-corrected fluorescence traces.
    """

    Fneu_demeaned = Fneu - np.median(Fneu, axis=1, keepdims=True)
    F_corrected = F - neucoeff * Fneu_demeaned
    if save_path is not None:
        np.save(save_path, F_corrected, allow_pickle=True)
    return F_corrected


def dFF(f, n_components=2):
    """
    Calculate ΔF/F using a Gaussian Mixture Model (GMM) to estimate baseline (F0).

    The baseline F0 is estimated as the mean of the lower component of a GMM
    fitted to the fluorescence distribution.

    Args:
        f (numpy.ndarray): Fluorescence traces (n_rois x n_frames).
        n_components (int, optional): Number of GMM components. Default 2.

    Returns:
        tuple: (dff, f0)
            - dff (np.ndarray): The calculated ΔF/F traces.
            - f0 (np.ndarray): The estimated baseline values (n_rois x 1).
    """
    f0 = np.zeros(f.shape[0])

    for i in tqdm(range(f.shape[0])):
        if np.all(np.isnan(f[i])):
            f0[i] = np.nan
        else:
            gmm = mixture.GaussianMixture(
                n_components=n_components, random_state=42
            ).fit(f[i].reshape(-1, 1))
            gmm_means = np.sort(gmm.means_[:, 0])
            f0[i] = gmm_means[0]
    f0 = f0.reshape(-1, 1)
    dff = (f - f0) / f0
    return dff, f0


def calculate_and_save_dFF(dpath, F, filename_suffix, n_components=2):
    """
    Calculate ΔF/F and save results to the Suite2p folder.

    Args:
        dpath (Path): Path to the Suite2p plane folder.
        F (numpy.ndarray): Neuropil-corrected fluorescence traces.
        filename_suffix (str): Suffix for the output filenames (e.g., '_ast').
        n_components (int, optional): Number of GMM components for baseline estimation.
            Default 2.

    Returns:
        tuple: (dff, f0)
            - dff (np.ndarray): The calculated ΔF/F traces.
            - f0 (np.ndarray): The estimated baseline values.
    """
    print("Calculating dF/F...")
    # Calculate dFFs and save to the suite2p folder
    print(f"n components for dFF calculation: {n_components}")
    dff, f0 = dFF(F, n_components=n_components)
    np.save(dpath / f"dff{filename_suffix}.npy", dff)
    np.save(dpath / f"f0{filename_suffix}.npy", f0)
    return dff, f0


def estimate_offset(datapath, n_components=3, save_path=None):
    """
    Estimate the optical offset for a session by fitting a GMM to a raw TIFF frame.

    Args:
        datapath (str or Path): Path to the folder containing raw ScanImage TIFFs.
        n_components (int, optional): Number of GMM components. Default 3.
        save_path (str or Path, optional): Path to save a diagnostic plot of the GMM fit.

    Returns:
        float: The estimated offset (mean of the lowest GMM component).
    """
    # find the first tiff at the path
    tiffs = list(Path(datapath).glob("*.tif"))
    if len(tiffs) == 0:
        raise ValueError(f"No tiffs found at {datapath}")
    tiff = tiffs[0]

    with TiffFile(tiff) as tf:
        # Load the first frame
        im = tf.asarray(key=0)

    # fit a gmm to the image pixels
    X = im.flatten().reshape(-1, 1)
    # Subset if the image is too large
    if X.shape[0] > 1000000:
        np.random.seed(42)
        X_subset = np.random.choice(X.flatten(), 1000000, replace=False).reshape(-1, 1)
    else:
        X_subset = X

    gmm = mixture.GaussianMixture(n_components=n_components, random_state=42).fit(
        X_subset
    )

    # find the lowest component
    offset = np.min(gmm.means_)

    if save_path is not None:
        sanity.plot_optical_offset_gmm(X_subset, gmm, offset, save_path=save_path)

    return offset


def correct_offset(datapath, offsets, first_frames, last_frames):
    """
    Subtract estimated offsets from concatenated fluorescence traces.

    Args:
        datapath (str or Path): Path to the .npy file containing concatenated traces.
        offsets (numpy.ndarray): Array of offsets, one per recording.
        first_frames (numpy.ndarray): Start frame indices for each recording.
        last_frames (numpy.ndarray): End frame indices for each recording.

    Returns:
        np.ndarray: The offset-corrected fluorescence traces.
    """
    # load the concatenated fluorescence trace
    F = np.load(datapath)
    # subtract offset for each recording
    for start, end, offset in zip(first_frames, last_frames, offsets):
        F[:, start:end] -= offset
    return F


def get_recording_frames(suite2p_dataset):
    """
    Get the frame boundaries for each recording in a concatenated Suite2p session.

    Args:
        suite2p_dataset (Dataset): Flexilims Dataset object for the Suite2p ROIs.

    Returns:
        tuple: (first_frames, last_frames)
            - first_frames (np.ndarray): Indices of the first frame for each recording.
            - last_frames (np.ndarray): Indices of the last frame for each recording.
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
        if not ops_path.exists():
            raise FileNotFoundError(
                f"Suite2p ops file not found at {ops_path}. "
                "Ensure that Suite2p processing completed successfully."
            )
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
