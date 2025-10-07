from functools import partial
import flexiznam as flz
import numpy as np
from pathlib import Path
from znamutils import slurm_it

from .calcium import process_concatenated_traces, split_recordings
from .calcium_s2p import reextract_masks

print = partial(print, flush=True)


@slurm_it(
    conda_env="2p-preprocess",
    slurm_options={
        "cpus-per-task": 1,
        "ntasks": 1,
        "time": "24:00:00",
        "mem": "256G",
        "partition": "ga100",
        "gres": "gpu:1",
    },
    module_list=["CUDA/12.1.1", "cuDNN/8.9.2.26-CUDA-12.1.1"],
    print_job_id=True,
)
def reextract_session(
    session, masks, flz_session=None, project=None, conflicts="abort"
):
    """Reextract masks and fluorescence traces for a session.

    This function creates a new suite2p dataset with the reextracted masks and traces.
    The new dataset is marked as annotated in flexilims.

    Args:
        session (str): name of the session
        masks (ndarray | str): Z x X x Y array of masks to be reextracted
        flz_session (Flexilims): flexilims session
        project (str, optional): name of the project. If not provided, it will be
            inferred from the flexilims session.
        conflicts (str, optional): defines behavior if recordings have already been
            reextracted. One of `abort`, `skip`, `append`, `overwrite`. Default `abort`

    Returns:
        ndarray: merged masks
    """
    print(f"Conflicts: {conflicts}")
    try:
        import suite2p
    except ImportError:
        raise ImportError(
            "suite2p is not installed. Please see 2p-preprocess ReadMe to install it"
        )
    if flz_session is None:
        assert project is not None, "Either flz_session or project must be provided"
        flz_session = flz.get_flexilims_session(project)
    elif project is not None:
        assert (
            flz_session.project == project
        ), "Provided project does not match the project of the provided flexilims session"

    if isinstance(masks, str) or isinstance(masks, Path):
        print(f"Loading masks from {masks}...")
        masks = load_mask(masks)
    assert isinstance(masks, np.ndarray), "masks must be a numpy array"
    if masks.ndim == 2:
        masks = masks[None, ...]  # add a plane dimension if only one plane
    elif masks.ndim != 3:
        raise ValueError(
            f"masks must be a 3D array (Z x X x Y), but got {masks.ndim}D array"
        )

    # get initial suite2p dataset
    suite2p_ds = flz.get_datasets(
        flexilims_session=flz_session,
        origin_name=session,
        dataset_type="suite2p_rois",
        exclude_datasets={"annotated": True},
        return_dataseries=True,
    )
    # remove datasets with annotated in the name
    suite2p_ds = suite2p_ds[~suite2p_ds["name"].str.contains("annotated")]
    assert (
        len(suite2p_ds) == 1
    ), f"Found {len(suite2p_ds)} non-annotated suite2p datasets for session {session}"
    suite2p_ds = flz.Dataset.from_dataseries(suite2p_ds.iloc[0], flz_session)

    # mark the original dataset as non annotated if needed
    online_label = suite2p_ds.extra_attributes.get("annotated", None)
    # is_labeled might be NaN, we need to check for non-equality to 'no'
    if online_label != False:
        suite2p_ds.extra_attributes["annotated"] = False
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
    suite2p_ds_annotated.extra_attributes["annotated"] = True

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
    new_fast_disk = (
        suite2p_ds_annotated.path / ops["fast_disk"].split(project)[1][1:] / "suite2p"
    )
    empty_planes = [not np.any(m) for m in masks]
    for iplane in range(len(masks)):
        if (fast_disk / f"plane{iplane}" / "data.bin").exists():
            continue
        if (new_fast_disk / f"plane{iplane}" / "data.bin").exists():
            continue

        # Check if there are masks in this plane
        if empty_planes[iplane]:
            print(f"No masks for plane {iplane}. Skipping")
            continue
        re_register = True

    # Update all attributes that are in ops by the ops value
    for key, value in ops.items():
        if isinstance(value, Path):
            value = str(value)
        if key in suite2p_ds_annotated.extra_attributes:
            ori = suite2p_ds_annotated.extra_attributes[key]
            if ori == value:
                continue
            print(f"Updating {key} from {ori} ({type(ori)}) to {value} ({type(value)})")
            suite2p_ds_annotated.extra_attributes[key] = value

    if (not re_register) and (suite2p_ds_annotated.flexilims_status != "not online"):
        print("suite2p_ds_annotated already online, no need to create ops")
    else:
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
                elif isinstance(v, str) and (v == "False"):
                    print(f"Found string False for {k}, converting to boolean")
                    ops[k] = False
                    suite2p_ds_annotated.extra_attributes[k] = False
                elif isinstance(v, str) and (v == "True"):
                    print(f"Found string True for {k}, converting to boolean")
                    ops[k] = True
                    suite2p_ds_annotated.extra_attributes[k] = True
                else:
                    ops[k] = v
            # Add the empty planes to the ignore_flyback field
            ops["ignore_flyback"] = list(np.where(empty_planes)[0])

            ops["keep_movie_raw"] = False
            ops["delete_bin"] = False
            ops["do_registration"] = 1
            # make a copy in the target directory
            target_dir = suite2p_ds_annotated.path_full / subdir.name
            target_dir.mkdir(exist_ok=True)
            # do_registration must > 1 to force redo, do it after saving the ops, since we
            # reload the ops only after the first call to s2p with roidetect = False
            if re_register:
                ops["do_registration"] = 2
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
        mini_ops["roidetect"] = False
        # run the registration but not detection else
        suite2p.run_s2p(ops=mini_ops)

        # Remove the force re-registration flag from ops
        for target_dir in suite2p_ds_annotated.path_full.iterdir():
            if not target_dir.name.startswith("plane"):
                continue
            planei = int(target_dir.name[5:])
            if empty_planes[planei]:
                continue
            if not target_dir.is_dir():
                continue
            ops = np.load(target_dir / "ops.npy", allow_pickle=True).item()
            # now we want to reextract the ROIs, so we set roidetect to True
            # It will not redo the detection if a stat.npy file is found
            ops["roidetect"] = True
            if ops["do_registration"] > 1:
                ops["do_registration"] = 1
            np.save(target_dir / "ops.npy", ops, allow_pickle=True)

    # Reextract masks and fluorescence traces
    # note that Lx,Ly and nplaces are read from the annotated ds, not from the ops
    # because we need the number of plane before we start iterating over planes
    print("Reextracting masks and fluorescence traces...")
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
        planes.append(stat[0]["iplane"])

    # Add empty plane 0 if it is not in the list of planes
    if 0 not in planes:
        print("No plane 0 found, adding empty plane 0")
        target_dir = suite2p_ds_annotated.path_full / "plane0"
        target_dir.mkdir(exist_ok=True)
        np.save(target_dir / "F.npy", np.array([[]]))
        np.save(target_dir / "Fneu.npy", np.array([[]]))
        np.save(target_dir / "stat.npy", np.array([]), allow_pickle=True)
        np.save(target_dir / "ops.npy", ops, allow_pickle=True)

    print("Calculating dF/F...")
    process_concatenated_traces(
        suite2p_dataset=suite2p_ds_annotated,
        ops=ops,
        project=project,
        flz_session=flz_session,
    )

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


def load_mask(path2mask):
    """Load masks from image, npy or stats file.

    Args:
        path2mask (str or Path): path to the mask file

    Returns:
        ndarray: Z x X x Y array of masks to be reextracted
    """
    path2mask = str(Path(path2mask).resolve())
    if path2mask.endswith(".npy"):
        return np.load(path2mask)
    elif (
        path2mask.endswith(".tif")
        or path2mask.endswith(".tiff")
        or path2mask.endswith(".png")
    ):
        from skimage.io import imread

        return imread(path2mask)
    else:
        raise ValueError(f"Unsupported mask file format: {path2mask}")
