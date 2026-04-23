from functools import partial
import os
import flexiznam as flz
import numpy as np

from .calcium_utils import estimate_offset, rolling_percentile

print = partial(print, flush=True)


def detrend(F, first_frames, last_frames, ops, fs):
    """
    Detrend fluorescence traces for each recording in a session.

    This function applies a rolling percentile filter to estimate the baseline
    of each ROI's fluorescence trace for each individual recording. The baseline
    is then either subtracted or divided out, relative to the first recording's baseline
    to maintain cross-recording consistency.

    Args:
        F (numpy.ndarray): Concatenated fluorescence traces (n_rois x n_frames).
        first_frames (numpy.ndarray): Start frame indices for each recording.
        last_frames (numpy.ndarray): End frame indices for each recording.
        ops (dict): Dictionary of settings, must include 'detrend_win' (window size in s),
            'detrend_pctl' (percentile), and 'detrend_method' ('subtract' or 'divide').
        fs (float): Sampling frequency in Hz.

    Returns:
        tuple: (F_detrended, all_rec_baseline)
            - F_detrended (np.ndarray): The detrended fluorescence traces.
            - all_rec_baseline (np.ndarray): The estimated baseline values.
    """
    win_frames = int(ops["detrend_win"] * fs)

    if win_frames % 2 == 0:
        pad_size = (win_frames // 2, win_frames // 2 - 1)
    else:
        pad_size = (win_frames // 2, win_frames // 2)  # Adjust for odd case

    all_rec_baseline = np.zeros_like(F)
    for i, (start, end) in enumerate(zip(first_frames, last_frames)):
        rec_rolling_baseline = np.zeros_like(F[:, start:end])
        for j in range(F.shape[0]):
            rolling_baseline = np.pad(
                rolling_percentile(
                    F[j, start:end],
                    win_frames,
                    ops["detrend_pctl"],
                ),
                pad_size,
                mode="edge",
            )

            rec_rolling_baseline[j, :] = rolling_baseline

        if i == 0:
            first_recording_baseline = np.median(rec_rolling_baseline, axis=1)
            first_recording_baseline = first_recording_baseline.reshape(-1, 1)
        if ops["detrend_method"] == "subtract":
            F[:, start:end] -= rec_rolling_baseline - first_recording_baseline
        else:
            F[:, start:end] /= rec_rolling_baseline / first_recording_baseline
        all_rec_baseline[:, start:end] = rec_rolling_baseline
    return F, all_rec_baseline


def estimate_offsets(suite2p_dataset, ops, project, flz_session):
    """
    Estimate optical offsets for all recordings associated with a Suite2p session.

    This function iterates through the raw data paths stored in the Suite2p dataset,
    identifies the original ScanImage TIFFs, and estimates the optical offset
    for each recording using a GMM.

    Args:
        suite2p_dataset (Dataset): Flexilims Dataset object for the Suite2p ROIs.
        ops (dict): Dictionary of preprocessing settings.
        project (str): Flexilims project name.
        flz_session (Flexilims): Active Flexilims session object.

    Returns:
        list: A list of estimated offsets, one per recording.
    """
    print("Estimating offsets...")

    offsets = []
    if not ops.get("correct_offset", True):
        print("Offset correction skipped.")
        offsets = [0] * len(suite2p_dataset.extra_attributes["data_path"])
        return offsets

    data_root = flz.get_data_root("raw", project, flz_session)
    for datapath in suite2p_dataset.extra_attributes["data_path"]:
        datapath = os.path.join(data_root, *str(datapath).split("/")[-4:])
        offsets.append(estimate_offset(datapath))
        print(f"Estimated offset for {datapath} is {offsets[-1]}")
        np.save(suite2p_dataset.path_full / "offsets.npy", offsets)
    return offsets
