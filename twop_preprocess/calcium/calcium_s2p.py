from functools import partial
from pathlib import Path
import flexiznam as flz
from flexiznam.schema import Dataset
import numpy as np

from ..utils import parse_si_metadata
from .calcium_utils import get_weights

print = partial(print, flush=True)


def spike_deconvolution_suite2p(suite2p_dataset, iplane, ops=None, ast_neuropil=True):
    """
    Run spike deconvolution on the concatenated recordings after ASt neuropil correction

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        iplane (int): which plane to run on
        ops (dict, optional): dictionary of suite2p settings
        ast_neuropil (bool, optional): wether to use Fast or Fstandard neuropil
            corrected trace for deconvolution. Default True.


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


def run_extraction(
    flz_session, project, session_name, conflicts, ops, delete_previous_run=False
):
    """
    Fetch data from flexilims and run suite2p with the provided settings

    Suite2p will generate a temporary folder, called suite2p where the initial binary
    are saved. It will save the actual output in the folder of the suite2p datasets.
    If conflict is overwrite, we re-run s2p.run but this function does not always redo
    everything is some files already exists. To start from a blank state, use
    delete_previous_run

    Args:
        flz_session (Flexilims): flexilims session
        project (str): name of the project, determines save path
        session_name (str): name of the session, used to find data on flexilims
        conflicts (str): defines behavior if recordings have already been split
        ops (dict): dictionary of suite2p settings
        delete_previous_run (bool): whether to delete previous runs. Default False


    Returns:
        Dataset: object containing the generated dataset

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
        if isinstance(v, datetime.datetime):
            ops[k] = v.strftime(r"%Y-%m-%d %H:%M:%S")
        if isinstance(v, np.float32):
            ops[k] = float(v)
        else:
            ops[k] = v

    suite2p_dataset.extra_attributes = ops
    suite2p_dataset.update_flexilims(mode="overwrite")
    return suite2p_dataset
