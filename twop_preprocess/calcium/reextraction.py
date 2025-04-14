from functools import partial
from pathlib import Path
import flexiznam as flz
import numpy as np


import warnings

from .calcium_processing import extract_dff
from .suite2p_interaction import spike_deconvolution_suite2p
from .postprocessing import split_recordings

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
    import suite2p
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
    import suite2p

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
