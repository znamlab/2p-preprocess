import numpy as np
import os
import re
import datetime
import flexiznam as flz
from flexiznam.schema import Dataset
from twop_preprocess.neuropil.ast_model import ast_model
import itertools
import suite2p
import suite2p.detection.anatomical
from suite2p.extraction import dcnv
from sklearn import mixture
from twop_preprocess.utils import parse_si_metadata, load_ops
from functools import partial
from tqdm import tqdm
from tifffile import TiffFile
from pathlib import Path
from numba import njit, prange
import matplotlib.pyplot as plt
from twop_preprocess.plotting_utils import sanity_check_utils as sanity
import warnings



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


def reextract_masks(masks, suite2p_ds):
    """
    Reextract masks from a suite2p dataset.

    Args:
        masks (ndarray): Z x X x Y array of masks to be reextracted
        suite2p_ds (Dataset): suite2p dataset

    Returns:
        merged_masks (ndarray): merged masks
        all_original_masks (list): list of original masks IDs corresponding to each
            plane
        all_F (list): list of F traces for each plane
        all_Fneu (list): list of Fneu traces
        all_stat (list): list of stats
        all_ops (list): list of ops

    """
    # There is case issue on some flexilims dataset, get the correct case first, but
    # try lower case if it fails
    Lx = int(suite2p_ds.extra_attributes.get("Lx", suite2p_ds.extra_attributes["lx"]))
    Ly = int(suite2p_ds.extra_attributes.get("Ly", suite2p_ds.extra_attributes["ly"]))
    nplanes = int(suite2p_ds.extra_attributes["nplanes"])

    # Calculate the number of rows and columns for the merged masks
    nX = np.ceil(np.sqrt(Ly * Lx * nplanes) / Lx)
    nX = int(nX)
    nY = np.ceil(nplanes / nX).astype(int)

    # Initialize outputs
    merged_masks = np.zeros((Ly * nY, Lx * nX))
    all_original_masks = []
    all_stat = []
    all_ops = []

    for iplane, masks_plane in enumerate(masks):
        if not np.any(masks_plane):
            print(f"No masks for plane {iplane}. Skipping")
            continue
        # get new mask values
        original_mask_values, reordered_masks = np.unique(
            masks_plane, return_inverse=True
        )
        # np.unique return_inverse returns the flattened array, so we need to reshape it
        reordered_masks = reordered_masks.reshape(masks_plane.shape).astype(int)

        # Add the reordered masks to the merged masks
        iX = iplane % nX
        iY = int(iplane / nX)
        merged_masks[iY * Ly : (iY + 1) * Ly, iX * Lx : (iX + 1) * Lx] = reordered_masks

        all_original_masks.append(original_mask_values[original_mask_values > 0])
        path2ops = suite2p_ds.path_full / f"plane{iplane}" / "ops.npy"
        ops = np.load(path2ops, allow_pickle=True).item()

        if iplane in ops["ignore_flyback"]:
            print(f"Skipping flyback plane {iplane}")
            continue

        # create ROIs stat
        stat = list(
            suite2p.detection.anatomical.masks_to_stats(
                reordered_masks, get_weights(ops)
            )
        )
        stat = suite2p.detection.roi_stats(
            stat,
            Ly,
            Lx,
            aspect=ops.get("aspect", None),
            diameter=ops.get("diameter", None),
            do_crop=ops.get("soma_crop", 1),
        )
        for i in range(len(stat)):
            stat[i]["iplane"] = iplane
        all_stat.append(stat)

        ops = suite2p.run_plane(ops, ops_path=str(path2ops.resolve()), stat=stat)
        all_ops.append(ops)
    save_folder = Path(ops["save_path0"]) / ops["save_folder"]
    return merged_masks, all_original_masks, all_stat, all_ops


def reextract_session(session, masks, flz_session, conflicts="abort"):
    """Reextract masks and fluorescence traces for a session.

    Args:
        session (str): name of the session
        masks (ndarray): Z x X x Y array of masks to be reextracted
        flz_session (Flexilims): flexilims session
        conflicts (str, optional): defines behavior if recordings have already been
            reextracted. One of `abort`, `skip`, `append`, `overwrite`. Default `abort`

    Returns:
        ndarray: merged masks
    """
    # get initial suite2p dataset
    suite2p_ds = flz.get_datasets(
        flexilims_session=flz_session,
        origin_name=session,
        dataset_type="suite2p_rois",
        exclude_datasets={"annotated": "yes"},
        return_dataseries=True,
    )
    # remove datasets with annotated in the name
    suite2p_ds = suite2p_ds[~suite2p_ds["name"].str.contains("annotated")]
    assert (
        len(suite2p_ds) == 1
    ), f"Found {len(suite2p_ds)} non-annotated suite2ßp datasets for session {session}"
    suite2p_ds = flz.Dataset.from_dataseries(suite2p_ds.iloc[0], flz_session)

    # mark the original dataset as non annotated if needed
    is_labeled = suite2p_ds.extra_attributes.get("annotated", "Not yet")
    # is_labeled might be NaN, we need to check for non-equality to 'no'
    if is_labeled != "no":
        suite2p_ds.extra_attributes["annotated"] = "no"
        suite2p_ds.update_flexilims(mode="update")

    # create or load a new suite2p dataset
    suite2p_ds_annotated = flz.Dataset.from_origin(
        origin_type="session",
        origin_name=session,
        dataset_type="suite2p_rois",
        conflicts=conflicts,
        flexilims_session=flz_session,
        verbose=True,
        base_name="suite2p_rois_annotated",
    )
    suite2p_ds_annotated.extra_attributes = suite2p_ds.extra_attributes
    # add a flag to the dataset to indicate that it is annotated
    suite2p_ds_annotated.extra_attributes["annotated"] = "yes"

    # handle conflicts
    target_dir = suite2p_ds_annotated.path_full / "combined"
    if target_dir.exists():
        if conflicts == "overwrite":
            print(f"{target_dir} already exists, overwriting!")
        elif conflicts == "skip":
            print(f"{target_dir} already exists, skipping!")
            return suite2p_ds_annotated
        else:
            raise ValueError(f"{target_dir} already exists, cannot append!")
    target_dir.mkdir(exist_ok=True, parents=True)

    # check if the binary still exist
    re_register = False
    # load one random ops file to get the project path
    ops = np.load(suite2p_ds.path_full / "plane0" / "ops.npy", allow_pickle=True).item()
    project = suite2p_ds.project
    fast_disk = flz.get_processed_path(project) / ops["fast_disk"].split(project)[1][1:]
    fast_disk /= "suite2p"
    empty_planes = [not np.any(m) for m in masks]
    for iplane in range(len(masks)):
        bin_file = fast_disk / f"plane{iplane}" / "data.bin"
        if bin_file.exists():
            continue
        # Check if there are masks in this plane
        if empty_planes[iplane]:
            print(f"No masks for plane {iplane}. Skipping")
            continue
        re_register = True

    # Copy ops to the target directory and check if binaries exist
    ori_path = str(suite2p_ds.path)
    new_path = str(suite2p_ds_annotated.path)
    for subdir in suite2p_ds.path_full.iterdir():
        if not subdir.name.startswith("plane"):
            continue
        planei = int(subdir.name[5:])
        if empty_planes[planei]:
            print(f"No masks for plane {planei}. Not creating ops")
            continue
        if not subdir.is_dir():
            continue
        source_ops_file = subdir / "ops.npy"
        if not source_ops_file.exists():
            continue

        # we replace all mentions to the original path with the new path
        ori_ops = np.load(source_ops_file, allow_pickle=True).item()
        ops = dict()
        for k, v in ori_ops.items():
            if isinstance(v, str) and (ori_path in v):
                ops[k] = v.replace(ori_path, new_path)
            elif isinstance(v, str) and (suite2p_ds.dataset_name in v):
                ops[k] = v.replace(
                    suite2p_ds.dataset_name, suite2p_ds_annotated.dataset_name
                )
            elif isinstance(v, Path) and (ori_path in str(v)):
                ops[k] = v.with_name(v.name.replace(ori_path, new_path))
            else:
                ops[k] = v
        # Add the empty planes to the ignore_flyback field
        ops["ignore_flyback"] = list(np.where(empty_planes)[0])

        # Always keep raw and bin files, manually delete if needed
        ops["keep_movie_raw"] = True
        ops["delete_bin"] = False

        # do_registration must > 1 to force redo
        if re_register:
            ops["do_registration"] = 2
        else:
            ops["do_registration"] = 1

        # make a copy in the target directory
        target_dir = suite2p_ds_annotated.path_full / subdir.name
        target_dir.mkdir(exist_ok=True)
        np.save(target_dir / "ops.npy", ops, allow_pickle=True)

    if re_register:  # We need to ensure the raw also exists
        print(f"Binary files not found. Force re-registration", flush=True)
        # we need to convert the tiff to binary to be able to re-register
        # copy from the last opened ops all the necessary fields which are identical for
        # all planes
        mini_ops = ops.copy()
        plane_dpt = [
            "meanImg",
            "save_path",
            "ops_path",
            "reg_file",
            "frames_per_folder",
            "nframes",
            "frames_per_file",
        ]
        for key in ops:
            if isinstance(ops[key], str) and "plane" in ops[key]:
                mini_ops.pop(key)
            elif key in plane_dpt:
                mini_ops.pop(key)
        ops = suite2p.io.tiff_to_binary(mini_ops)

    # Reextract masks and fluorescence traces
    (
        merged_masks,
        all_original_masks,
        all_stat,
        all_ops,
    ) = reextract_masks(masks.astype(int), suite2p_ds_annotated)

    # Save the mask correspondance for each plane and new mask stats
    np.save(
        target_dir / "stat.npy", np.concatenate(all_stat, axis=0), allow_pickle=True
    )
    # mask for each plane have different length, cannot save a single array
    np.savez(
        suite2p_ds_annotated.path_full / "original_masks.npz",
        **{f"plane{i}": mask for i, mask in enumerate(all_original_masks)},
    )
    np.save(suite2p_ds_annotated.path_full / "merged_masks.npy", merged_masks)

    planes = []
    for stat, ops in zip(all_stat, all_ops):
        target_dir = suite2p_ds_annotated.path_full / f"plane{stat[0]['iplane']}"
        target_dir.mkdir(exist_ok=True)
        spike_deconvolution_suite2p(
            suite2p_ds_annotated, stat[0]["iplane"], ops, ast_neuropil=False
        )
        planes.append(stat[0]["iplane"])

    # Add empty plane 0 if it is not in the list of planes
    if 0 not in planes:
        print("No plane 0 found, adding empty plane 0")
        target_dir = suite2p_ds_annotated.path_full / "plane0"
        target_dir.mkdir(exist_ok=True)
        np.save(target_dir / "F.npy", np.array([[]]))
        np.save(target_dir / "Fneu.npy", np.array([[]]))
        np.save(target_dir / "spks.npy", np.array([[]]))
        np.save(target_dir / "stat.npy", np.array([]), allow_pickle=True)
        np.save(target_dir / "ops.npy", ops)

    print("Calculating dF/F...")
    extract_dff(suite2p_ds_annotated, ops)

    print("Splitting recordings...")
    split_recordings(
        flz_session,
        suite2p_ds_annotated,
        conflicts="overwrite",
        base_name="suite2p_traces_annotated",
        extra_attributes=dict(annotated=True),
    )

    print("Updating flexilims...")
    suite2p_ds_annotated.update_flexilims(mode="update")
    return suite2p_ds_annotated


def run_extraction(flz_session, project, session_name, conflicts, ops):
    """
    Fetch data from flexilims and run suite2p with the provided settings

    Args:
        flz_session (Flexilims): flexilims session
        project (str): name of the project, determines save path
        session_name (str): name of the session, used to find data on flexilims
        conflicts (str): defines behavior if recordings have already been split
        ops (dict): dictionary of suite2p settings

    Returns:
        Dataset: object containing the generated dataset

    """
    # get experimental session
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flz_session
    )
    if exp_session is None:
        raise ValueError(f"Session {session_name} not found on flexilims")

    # fetch an existing suite2p dataset or create a new suite2p dataset
    if conflicts == "overwrite":
        suite2p_datasets = flz.get_datasets(
            origin_name=session_name,
            dataset_type="suite2p_rois",
            project_id=project,
            flexilims_session=flz_session,
            return_dataseries=False,
        )
        if len(suite2p_datasets) == 0:
            raise ValueError(
                f"No suite2p dataset found for session {session_name}. Cannot overwrite."
            )
        elif len(suite2p_datasets) > 1:
            print(
                f"{len(suite2p_datasets)} suite2p datasets found for session {session_name}"
            )
            print("Overwriting the last one...")
            suite2p_dataset = suite2p_datasets[
                np.argmax(
                    [
                        datetime.datetime.strptime(i.created, "%Y-%m-%d %H:%M:%S")
                        for i in suite2p_datasets
                    ]
                )
            ]
        else:
            suite2p_dataset = suite2p_datasets[0]

    else:
        suite2p_dataset = Dataset.from_origin(
            project=project,
            origin_type="session",
            origin_id=exp_session["id"],
            dataset_type="suite2p_rois",
            conflicts=conflicts,
        )
    # if already on flexilims and not re-processing, then do nothing
    if (suite2p_dataset.get_flexilims_entry() is not None) and conflicts == "skip":
        print(
            "Session {} already processed... skipping extraction...".format(
                exp_session["name"]
            )
        )
        return suite2p_dataset

    # fetch SI datasets
    si_datasets = flz.get_datasets_recursively(
        origin_id=exp_session["id"],
        parent_type="recording",
        filter_parents={"recording_type": "two_photon"},
        dataset_type="scanimage",
        flexilims_session=flz_session,
        return_paths=True,
    )
    datapaths = []
    for _, p in si_datasets.items():
        datapaths.extend(p)
    # set save path
    ops["save_path0"] = str(suite2p_dataset.path_full.parent)
    ops["save_folder"] = suite2p_dataset.dataset_name
    # assume frame rates are the same for all recordings
    si_metadata = parse_si_metadata(datapaths[0])
    ops["fs"] = si_metadata["SI.hRoiManager.scanVolumeRate"]
    if si_metadata["SI.hStackManager.enable"]:
        ops["nplanes"] = si_metadata["SI.hStackManager.numSlices"]
    else:
        ops["nplanes"] = 1
    # calculate cell diameter based on zoom and pixel size
    ops["diameter"] = int(
        round(
            si_metadata["SI.hRoiManager.pixelsPerLine"]
            * si_metadata["SI.hRoiManager.scanZoomFactor"]
            * ops["diameter_multiplier"]
        )
    )
    # print ops
    print("Running suite2p with the following ops:")
    for k, v in ops.items():
        print(f"{k}: {v}")
    # run suite2p
    db = {"data_path": datapaths}
    opsEnd = suite2p.run_s2p(ops=ops, db=db)
    if "date_proc" in opsEnd:
        opsEnd["date_proc"] = opsEnd["date_proc"].isoformat()
    # update the database
    ops = ops.copy()
    for k, v in opsEnd.items():
        if isinstance(v, np.ndarray):
            print(f"{k} is a numpy array, skipping")
            continue
        if isinstance(v, datetime.datetime):
            ops[k] = v.strftime(r"%Y-%m-%d %H:%M:%S")
        if isinstance(v, np.float32):
            ops[k] = float(v)
        else:
            ops[k] = v

    suite2p_dataset.extra_attributes = ops
    suite2p_dataset.update_flexilims(mode="overwrite")
    return suite2p_dataset
        
        
def extract_dff(suite2p_dataset, ops, project, flz_session):
    """
    Correct offsets, detrend, calculate dF/F and deconvolve spikes for the whole session.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        ops (dict): dictionary of suite2p settings

    """
    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    offsets = []
    for datapath in suite2p_dataset.extra_attributes["data_path"]:
        datapath = os.path.join(flz.get_data_root('raw', project, flz_session), 
                                *datapath.split('/')[-4:]) # add the raw path from flexiznam config
        if ops["correct_offset"]:
            offsets.append(estimate_offset(datapath))
            print(f"Estimated offset for {datapath} is {offsets[-1]}")
            np.save(suite2p_dataset.path_full / "offsets.npy", offsets)
        else:
            offsets.append(0)

    fs = suite2p_dataset.extra_attributes["fs"]
    # run neuropil correction, dFF calculation and spike deconvolution
    for iplane in range(int(suite2p_dataset.extra_attributes["nplanes"])):
        dpath = suite2p_dataset.path_full / f"plane{iplane}"
        F = np.load(dpath / "F.npy")
        if F.shape[1] == 0:
            print(f"No rois found for plane {iplane}")
            continue
        Fneu = np.load(dpath / "Fneu.npy")
        if ops["sanity_plots"]:
            os.makedirs(dpath / "sanity_plots", exist_ok=True)
            np.random.seed(0)
            random_rois = np.random.choice(F.shape[0], ops["plot_nrois"], replace=False)
            sanity.plot_raw_trace(F, random_rois, Fneu)
            plt.savefig(dpath / "sanity_plots/raw_trace.png")
        F = correct_offset(
            dpath / "F.npy", offsets, first_frames[:, iplane], last_frames[:, iplane]
        )
        Fneu = correct_offset(
            dpath / "Fneu.npy", offsets, first_frames[:, iplane], last_frames[:, iplane]
        )
        if ops["detrend"]:
            print("Detrending...")
            F_offset_corrected = F.copy()
            Fneu_offset_corrected = Fneu.copy()
            F, F_trend = detrend(
                F, first_frames[:, iplane], last_frames[:, iplane], ops, fs
            )
            Fneu, Fneu_trend = detrend(
                Fneu, first_frames[:, iplane], last_frames[:, iplane], ops, fs
            )

        if ops["ast_neuropil"]:
            print("Running ASt neuropil correction...")
            correct_neuropil(dpath, F, Fneu)
            Fast = np.load(dpath / "Fast.npy")
            dff, f0 = calculate_dFF(dpath, Fast, Fneu, ops)
            print("Deconvolve spikes from neuropil corrected trace...")
            spike_deconvolution_suite2p(suite2p_dataset, iplane, ops)
        else:
            dff, f0 = calculate_dFF(dpath, F, Fneu, ops)
            Fast = np.zeros_like(F)

        if ops["sanity_plots"]:
            sanity.plot_raw_trace(F_offset_corrected, random_rois, Fneu)
            plt.savefig(dpath / "sanity_plots/offset_corrected.png")
            sanity.plot_dff(Fast, dff, f0, random_rois)
            plt.savefig(dpath / "sanity_plots" / f'dffs_n{ops["dff_ncomponents"]}.png')
            sanity.plot_fluorescence_matrices(F, Fneu, Fast, dff, ops["neucoeff"])
            plt.savefig(dpath / "sanity_plots" / f"fluorescence_matrices.png")
            if ops["detrend"]:
                sanity.plot_detrended_trace(
                    F_offset_corrected,
                    F_trend,
                    F,
                    Fneu_offset_corrected,
                    Fneu_trend,
                    Fneu,
                    random_rois,
                )
                plt.savefig(dpath / "sanity_plots" / "detrended.png")
            if ops["ast_neuropil"]:
                sanity.plot_raw_trace(F, random_rois, Fast, titles=["F", "Fast"])
                plt.savefig(dpath / "sanity_plots" / "neuropil_corrected.png")


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


@njit(parallel=True)
def rolling_percentile(arr, window, percentile):
    output = np.empty(len(arr) - window + 1)
    for i in prange(len(output)):
        output[i] = np.percentile(arr[i : i + window], percentile)
    return output


def detrend(F, first_frames, last_frames, ops, fs):
    """
    Detrend the concatenated fluorescence trace for each recording.

    Args:
        F (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois extracted from suite2p
        first_frames (numpy.ndarray): shape nrecordings, first frame of each recording
        last_frames (numpy.ndarray): shape nrecordings, last frame of each recording
        ops (dict): dictionary of suite2p settings

    Returns:
        F (numpy.ndarray): shape nrois x time, detrended fluorescence trace for all rois extracted from suite2p

    """
    win_frames = int(ops["detrend_win"] * fs)

    if win_frames % 2 == 0:
        pad_size = (win_frames // 2, win_frames // 2 - 1)
    else:
        pad_size = (win_frames // 2 , win_frames // 2 )  # Adjust for odd case

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
                mode='edge',
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


def correct_neuropil(dpath, Fr, Fn):
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


def dFF(f, n_components=2):
    """
    Helper function for calculating dF/F from raw fluorescence trace.
    Args:
        f (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois extracted from suite2p
        n_components (int): number of components for GMM. default 2.

    Returns:
        dffs (numpy.ndarray): shape nrois x time, dF/F for all rois extracted from suite2p

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


def calculate_dFF(dpath, F, Fneu, ops):
    """
    Calculate dF/F for the whole session with concatenated recordings after neuropil correction.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        iplane (int): which plane.
        n_components (int): number of components for GMM. default 2.
        ast_neuropil (bool): whether to use ASt neuropil correction or not. Default True.
        neucoeff (float): coefficient for neuropil correction. Only used if ast_neuropil
            is False. Default 0.7.

    """
    print("Calculating dF/F...")
    if not ops["ast_neuropil"]:
        F = F - ops["neucoeff"] * Fneu
    # Calculate dFFs and save to the suite2p folder
    print(f"n components for dFF calculation: {ops['dff_ncomponents']}")
    dff, f0 = dFF(F, n_components=ops["dff_ncomponents"])
    np.save(dpath / "dff_ast.npy" if ops["ast_neuropil"] else dpath / "dff.npy", dff)
    np.save(dpath / "f0_ast.npy" if ops["ast_neuropil"] else dpath / "f0.npy", f0)
    return dff, f0


def spike_deconvolution_suite2p(suite2p_dataset, iplane, ops={}, ast_neuropil=True):
    """
    Run spike deconvolution on the concatenated recordings after ASt neuropil correction

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        iplane (int): which plane to run on
        ops (dict): dictionary of suite2p settings

    """
    # Load the Fast.npy file and ops.npy file
    if ast_neuropil:
        F_path = suite2p_dataset.path_full / f"plane{iplane}" / "Fast.npy"
        spks_path = suite2p_dataset.path_full / f"plane{iplane}" / "spks_ast.npy"
    else:
        F_path = suite2p_dataset.path_full / f"plane{iplane}" / "F.npy"
        spks_path = suite2p_dataset.path_full / f"plane{iplane}" / "spks.npy"

    suite2p_ops_path = suite2p_dataset.path_full / f"plane{iplane}" / "ops.npy"
    F = np.load(F_path)
    suite2p_ops = np.load(suite2p_ops_path, allow_pickle=True).tolist()
    suite2p_ops.update(ops)
    ops = suite2p_ops

    # baseline operation
    F = dcnv.preprocess(
        F=F,
        baseline=ops["baseline_method"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
    )

    # get spikes
    spks = dcnv.oasis(F=F, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
    np.save(spks_path, spks)


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
    nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))
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


def split_recordings(
    flz_session,
    suite2p_dataset,
    conflicts,
    base_name="suite2p_traces",
    extra_attributes=None,
):
    """
    suite2p concatenates all the recordings in a given session into a single file.
    To facilitate downstream analyses, we cut them back into chunks and add them
    to flexilims as children of the corresponding recordings.

    Args:
        flz_session (Flexilims): flexilims session
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        conflicts (str): defines behavior if recordings have already been split
        base_name (str, optional): base name for the split datasets. Default
            "suite2p_traces"
        extra_attributes (dict, optional): Extra attributes to add to the split datasets
            on flexilims. Used only for identification. Default None.

    """
    # get scanimage datasets
    datasets = flz.get_datasets_recursively(
        origin_id=suite2p_dataset.origin_id,
        parent_type="recording",
        filter_parents={"recording_type": "two_photon"},
        dataset_type="scanimage",
        flexilims_session=flz_session,
        return_paths=True,
    )
    datapaths = []
    recording_ids = []
    frame_rates = []
    for recording, paths in datasets.items():
        datapaths.extend(paths)
        recording_ids.extend(itertools.repeat(recording, len(paths)))
        frame_rates.extend(
            [
                parse_si_metadata(this_path)["SI.hRoiManager.scanVolumeRate"]
                for this_path in paths
            ]
        )

    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))

    datasets_out = []
    for raw_datapath, recording_id, first_frames_rec, last_frames_rec in zip(
        datapaths, recording_ids, first_frames, last_frames
    ):
        # get the path for split dataset
        split_dataset = flz.Dataset.from_origin(
            origin_id=recording_id,
            dataset_type="suite2p_traces",
            base_name=base_name,
            flexilims_session=flz_session,
            conflicts=conflicts,
        )
        # Set the extra_attributes to match that of suite2p
        # minimum number of frames across planes
        nframes = np.min(last_frames_rec - first_frames_rec)
        si_metadata = parse_si_metadata(raw_datapath)
        split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
        split_dataset.extra_attributes["fs"] = si_metadata[
            "SI.hRoiManager.scanVolumeRate"
        ]
        split_dataset.extra_attributes["nframes"] = nframes
        if extra_attributes is not None:
            split_dataset.extra_attributes.update(dict(extra_attributes))
        if (split_dataset.flexilims_status() == "up-to-date") and (conflicts == "skip"):
            print(f"Dataset {split_dataset.dataset_name} already exists... skipping...")
            continue

        split_dataset.path_full.mkdir(parents=True, exist_ok=True)
        np.save(split_dataset.path_full / "si_metadata.npy", si_metadata)
        # load processed data
        for iplane, start in zip(range(nplanes), first_frames_rec):
            suite2p_path = suite2p_dataset.path_full / f"plane{iplane}"
            split_path = split_dataset.path_full / f"plane{iplane}"
            try:
                split_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f"Error creating directory {split_path}")
            F = np.load(suite2p_path / "F.npy")
            if F.shape[1] == 0:
                print(f"No rois found when splitting recordings for plane {iplane}")
                continue
            Fneu, spks = (
                np.load(suite2p_path / "Fneu.npy"),
                np.load(suite2p_path / "spks.npy"),
            )
            end = start + nframes
            np.save(split_path / "F.npy", F[:, start:end])
            np.save(split_path / "Fneu.npy", Fneu[:, start:end])
            np.save(split_path / "spks.npy", spks[:, start:end])
            if suite2p_dataset.extra_attributes["ast_neuropil"]:
                Fast, dff_ast, spks_ast = (
                    np.load(suite2p_path / "Fast.npy"),
                    np.load(suite2p_path / "dff_ast.npy"),
                    np.load(suite2p_path / "spks_ast.npy"),
                )
                np.save(split_path / "Fast.npy", Fast[:, start:end])
                np.save(split_path / "dff_ast.npy", dff_ast[:, start:end])
                np.save(split_path / "spks_ast.npy", spks_ast[:, start:end])
            else:
                dff = np.load(suite2p_path / "dff.npy")
                np.save(split_path / "dff.npy", dff[:, start:end])
        split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
        split_dataset.extra_attributes["fs"] = si_metadata[
            "SI.hRoiManager.scanVolumeRate"
        ]
        split_dataset.extra_attributes["nframes"] = nframes
        split_dataset.update_flexilims(mode="overwrite")
        datasets_out.append(split_dataset)
    return datasets_out


def extract_session(
    project,
    session_name,
    conflicts=None,
    run_split=False,
    run_suite2p=True,
    run_dff=True,
    ops={},
):
    """
    Process all the 2p datasets for a given session

    Args:
        project (str): name of the project, e.g. '3d_vision'
        session_name (str): name of the session
        conflicts (str): how to treat existing processed data
        run_split (bool): whether or not to run splitting for different folders
        run_suite2p (bool): whether or not to run extraction
        run_dff (bool): whether or not to run dff calculation

    """
    # get session info from flexilims
    print("Connecting to flexilims...")
    flz_session = flz.get_flexilims_session(project)
    ops = load_ops(ops)
    if run_suite2p:
        suite2p_dataset = run_extraction(
            flz_session, project, session_name, conflicts, ops
        )
    else:
        suite2p_datasets = flz.get_datasets(
            origin_name=session_name,
            dataset_type="suite2p_rois",
            project_id=project,
            flexilims_session=flz_session,
            return_dataseries=False,
        )
        if len(suite2p_datasets) == 0:
            raise ValueError(f"No suite2p dataset found for session {session_name}")
        elif len(suite2p_datasets) > 1:
            print(
                f"{len(suite2p_datasets)} suite2p datasets found for session {session_name}"
            )
            print("Splitting the last one...")
            suite2p_dataset = suite2p_datasets[
                np.argmax(
                    [
                        datetime.datetime.strptime(i.created, "%Y-%m-%d %H:%M:%S")
                        for i in suite2p_datasets
                    ]
                )
            ]
        else:
            suite2p_dataset = suite2p_datasets[0]

    if run_dff:
        print("Calculating dF/F...")
        extract_dff(suite2p_dataset, ops, project, flz_session)

    if run_split:
        print("Splitting recordings...")
        split_recordings(flz_session, suite2p_dataset, conflicts=conflicts)
