from functools import partial
import flexiznam as flz
import numpy as np

from .calcium_utils import estimate_offset, rolling_percentile

print = partial(print, flush=True)


def detrend(F, first_frames, last_frames, ops, fs):
    """
    Detrend the concatenated fluorescence trace for each recording.

    Args:
        F (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois
            extracted from suite2p
        first_frames (numpy.ndarray): shape nrecordings, first frame of each recording
        last_frames (numpy.ndarray): shape nrecordings, last frame of each recording
        ops (dict): dictionary of suite2p settings

    Returns:
        F (numpy.ndarray): shape nrois x time, detrended fluorescence trace for all rois
            extracted from suite2p

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
    Estimate the offsets for each recording in the session.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        ops (dict): dictionary of suite2p settings
        project (str): name of the project
        flz_session (Flexilims): flexilims session

    Returns:
        list: list of offsets for each recording
    """
    print("Estimating offsets...")

    offsets = []
    if not ops.get("correct_offset", True):
        print("Offset correction skipped.")
        offsets = [0] * len(suite2p_dataset.extra_attributes["data_path"])
        return offsets

    data_root = flz.get_data_root("raw", project, flz_session)
    for datapath in suite2p_dataset.extra_attributes["data_path"]:
        datapath = os.path.join(data_root, *datapath.split("/")[-4:])
        offsets.append(estimate_offset(datapath))
        print(f"Estimated offset for {datapath} is {offsets[-1]}")
        np.save(suite2p_dataset.path_full / "offsets.npy", offsets)
    return offsets
