from functools import partial
from pathlib import Path
import flexiznam as flz
from flexiznam.schema import Dataset
import numpy as np

from ..utils import parse_si_metadata
from .calcium_utils import get_weights
import datetime

print = partial(print, flush=True)


def spike_deconvolution_suite2p(suite2p_dataset, iplane, ops=None, ast_neuropil=True):
    """
    Run spike deconvolution on the concatenated recordings after ASt neuropil correction.

    This function uses Suite2p's OASIS implementation to extract spikes from
    neuropil-corrected fluorescence traces. It can use either ASt-corrected
    traces (Fast.npy) or standard-corrected traces (Fstandard.npy).

    Args:
        suite2p_dataset (Dataset): Flexilims Dataset object containing concatenated recordings.
        iplane (int): The plane index to process.
        ops (dict, optional): Dictionary of Suite2p settings. If None, uses the
            settings stored in the dataset's ops.npy.
        ast_neuropil (bool, optional): Whether to use the ASt-corrected trace
            (`Fast.npy`) or the standard trace (`Fstandard.npy`) for deconvolution.
            Default True.
    """
    try:
        from suite2p.extraction import dcnv
    except ImportError:
        raise ImportError(
            "suite2p is not installed. Please see 2p-preprocess ReadMe to install it"
        )

    if ops is None:
        ops = {}
    # Load the Fast.npy file and ops.npy file
    if ast_neuropil:
        F_path = suite2p_dataset.path_full / f"plane{iplane}" / "Fast.npy"
        spks_path = suite2p_dataset.path_full / f"plane{iplane}" / "spks_ast.npy"
    else:
        F_path = suite2p_dataset.path_full / f"plane{iplane}" / "Fstandard.npy"
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


def reextract_masks(masks, suite2p_ds):
    """
    Re-extract fluorescence traces from registered binaries using a new set of masks.

    This function takes an existing Suite2p dataset and a new set of ROI masks,
    calculates ROI statistics (weights, overlap, etc.), and runs the Suite2p
    extraction pipeline to generate new fluorescence (F) and neuropil (Fneu) traces.

    Args:
        masks (np.ndarray): 3D array of masks (Z x Ly x Lx) where each non-zero value
            represents a unique ROI ID.
        suite2p_ds (Dataset): Flexilims Dataset object for the existing Suite2p results.

    Returns:
        tuple: (merged_masks, all_original_masks, all_stats, all_ops)
            - merged_masks (np.ndarray): 2D array of all plane masks tiled into one image.
            - all_original_masks (list): List of unique ROI IDs found in each plane.
            - all_stats (list): List of Suite2p stats dictionaries for the new ROIs.
            - all_ops (list): List of updated Suite2p ops for each plane.
    """
    try:
        import suite2p
    except ImportError:
        raise ImportError(
            "suite2p is not installed. Please see 2p-preprocess ReadMe to install it"
        )
    import suite2p.detection.anatomical

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
    all_stats = []
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
        weights = get_weights(ops)
        # weights in cropped Lyc x Lxc arrays, 0-pad to Ly x Lx
        xrange, yrange = ops["xrange"], ops["yrange"]
        weights = np.pad(
            weights,
            ((yrange[0], Ly - yrange[1]), (xrange[0], Lx - xrange[1])),
            "constant",
        )
        np.nan_to_num(weights, copy=False)

        stats = suite2p.detection.anatomical.masks_to_stats(reordered_masks, weights)
        stats = suite2p.detection.roi_stats(
            stats,
            Ly,
            Lx,
            aspect=ops.get("aspect", None),
            diameter=ops.get("diameter", None),
            do_crop=ops.get("soma_crop", 1),
            max_overlap=ops["max_overlap"],
        )
        for i in range(len(stats)):
            stats[i]["iplane"] = iplane
        all_stats.append(stats)

        ops_s2p = suite2p.run_plane(ops, ops_path=str(path2ops.resolve()), stat=stats)
        if np.any(ops_s2p["yrange"] != ops["yrange"]) or np.any(
            ops_s2p["xrange"] != ops["xrange"]
        ):
            print("Updating Vcorr based on new registration range")
            # that works only for anatomical_only = 3
            if ops_s2p["anatomical_only"] != 3:
                raise NotImplementedError(
                    "Reextraction of masks only implemented for anatomical_only=3"
                )
            ops_s2p["Vcorr"] = ops_s2p["meanImgE"][
                ops_s2p["yrange"][0] : ops_s2p["yrange"][1],
                ops_s2p["xrange"][0] : ops_s2p["xrange"][1],
            ]
        ops_s2p["weight_image"] = weights
        # save modified ops
        np.save(path2ops, ops_s2p, allow_pickle=True)
        all_ops.append(ops_s2p)

    return merged_masks, all_original_masks, all_stats, all_ops


def run_extraction(
    flz_session, project, session_name, conflicts, ops, delete_previous_run=False
):
    """
    Fetch data from Flexilims and run Suite2p ROI extraction.

    This function identifies ScanImage datasets on Flexilims for the given session,
    configures the Suite2p `ops` dictionary (including frame rate and cell diameter
    calculations based on ScanImage metadata), and runs the Suite2p pipeline.
    Results are saved to the project's processed data path and registered on Flexilims.

    Args:
        flz_session (Flexilims): Active Flexilims session object.
        project (str): Flexilims project name.
        session_name (str): Name of the experimental session.
        conflicts (str): Behavior if a Suite2p dataset already exists
            ('skip', 'overwrite', 'abort').
        ops (dict): Dictionary of Suite2p settings.
        delete_previous_run (bool, optional): If True, deletes existing Suite2p binary files
            and output before starting. Default False.

    Returns:
        Dataset: The newly created (or existing) Flexilims Suite2p ROIs Dataset.
    """
    import suite2p

    # get experimental session
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flz_session
    )
    if exp_session is None:
        raise ValueError(f"Session {session_name} not found on flexilims")

    # fetch an existing suite2p dataset or create a new suite2p dataset
    suite2p_dataset = Dataset.from_origin(
        project=project,
        origin_type="session",
        origin_id=exp_session["id"],
        dataset_type="suite2p_rois",
        conflicts=conflicts,
    )
    # TODO: If there is more than one suite2p_rois `overwrite` will crash. Previous
    #  code would overwrite the last one. Decide what to do

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

    # Optionally delete everythin
    if delete_previous_run:
        # Check if output folder is empty
        output_folder = Path(ops["save_path0"])
        tmp_folder = output_folder / "suite2p"
        if tmp_folder.exists():
            # Delete content
            for item in tmp_folder.rglob("*.bin"):
                item.unlink()
        save_folder = output_folder / ops["save_folder"]
        if save_folder.exists():
            for npy_file in save_folder.rglob("*.npy"):
                npy_file.unlink()

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
        if isinstance(v, datetime.datetime):  # noqa: F821
            ops[k] = v.strftime(r"%Y-%m-%d %H:%M:%S")
        else:
            ops[k] = v

    suite2p_dataset.extra_attributes = ops
    suite2p_dataset.update_flexilims(mode="overwrite")
    return suite2p_dataset
