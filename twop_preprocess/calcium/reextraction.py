from functools import partial
import flexiznam as flz
import numpy as np
from pathlib import Path
from znamutils import slurm_it
from skimage.measure import label
from skimage.io import imsave
from matplotlib import pyplot as plt
from .calcium import process_concatenated_traces, split_recordings
from .calcium_s2p import reextract_masks
from ..plotting_utils import s2p_plotting_utils

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
    session,
    masks,
    flz_session=None,
    project=None,
    conflicts="abort",
    attribute_changes=None,
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
        attribute_changes (dict, optional): dictionary of attributes to change in the
            annotated dataset compared to the original. Default None

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
    if attribute_changes is not None:
        suite2p_ds_annotated.extra_attributes.update(attribute_changes)
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
    fast_disk = Path(ops["fast_disk"])
    if not fast_disk.exists():
        print(f"Fast disk does not exist, has project root changed? {fast_disk}")
        fast_disk = (
            flz.get_processed_path(project) / str(fast_disk).split(project)[1][1:]
        )
        print(f"Replacing with {fast_disk}")

    empty_planes = [not np.any(m) for m in masks]
    print(f"Looking for data in {fast_disk}")
    for iplane in range(len(masks)):
        if (fast_disk / "suite2p" / f"plane{iplane}" / "data.bin").exists():
            print(
                f"Data found in {fast_disk / 'suite2p' / f'plane{iplane}' / 'data.bin'}"
            )
            continue
        # Check if there are masks in this plane
        if empty_planes[iplane]:
            print(f"No masks for plane {iplane}. Skipping")
            continue
        print(f"No data for plane {iplane}. Re-registering")
        re_register = True

    # Update all attributes that are in ops by the ops value
    rewrite_ops = False
    for key, value in ops.items():
        if isinstance(value, Path):
            value = str(value)
        if key in suite2p_ds_annotated.extra_attributes:
            ori = suite2p_ds_annotated.extra_attributes[key]
            if ori == value:
                continue
            if key in attribute_changes:
                print(f"Keeping {key} different ({ori} instead of {value})")
                rewrite_ops = True
                continue
            print(f"Updating {key} from {ori} ({type(ori)}) to {value} ({type(value)})")
            suite2p_ds_annotated.extra_attributes[key] = value

    folder_exists = suite2p_ds_annotated.path_full.exists()
    # For debug
    print("\n")
    print(f"Dataset name: {suite2p_ds_annotated.full_name}")
    print(f"Folder exists: {folder_exists}")
    flm_stat = suite2p_ds_annotated.flexilims_status()
    print(f"Flexilims status: {flm_stat}")
    is_online = flm_stat != "not online"
    if (not re_register) and folder_exists and is_online and not rewrite_ops:
        print("suite2p_ds_annotated already online, no need to create ops")
        print("\n")
    else:
        # Copy ops to the target directory and check if binaries exist
        ori_path = str(suite2p_ds.path_full)
        new_path = str(suite2p_ds_annotated.path_full)
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
                if k in attribute_changes:
                    ops[k] = attribute_changes[k]
                    print(f"Setting new ops file with {k}={attribute_changes[k]}")
                elif isinstance(v, str) and (ori_path in v):
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

            print(f"\nSource file {source_ops_file}:")
            for k in [
                "save_path0",
                "save_path",
                "save_folder",
                "fast_disk",
                "ops_path",
                "reg_file",
            ]:
                print(f"{k}: {ops[k]}")

            ops["fast_disk"] = str(fast_disk)
            # Add the empty planes to the ignore_flyback field
            ops["ignore_flyback"] = list(np.where(empty_planes)[0])

            ops["keep_movie_raw"] = False
            ops["delete_bin"] = False
            ops["do_registration"] = 1
            # make a copy in the target directory
            target_dir = suite2p_ds_annotated.path_full / subdir.name
            ops_path = target_dir / "ops.npy"
            assert ops["ops_path"] == str(ops_path)
            target_dir.mkdir(exist_ok=True)
            # do_registration must > 1 to force redo, do it after saving the ops, since we
            # reload the ops only after the first call to s2p with roidetect = False
            if re_register:
                ops["do_registration"] = 2
            np.save(ops_path, ops, allow_pickle=True)

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

        # Verifying registered binaries
        print("Verifying registered binaries...")
        for iplane in range(len(masks)):
            if empty_planes[iplane]:
                continue
            bin_path = (
                suite2p_ds_annotated.path_full.parent
                / "suite2p"
                / f"plane{iplane}"
                / "data.bin"
            )
            if not bin_path.exists():
                raise FileNotFoundError(
                    f"Registration failed to produce binary for plane {iplane} at {bin_path}"
                )
            if bin_path.stat().st_size == 0:
                raise ValueError(
                    f"Registration produced an empty binary for plane {iplane} at {bin_path}"
                )

        # Remove the force re-registration flag from ops
        for subdir in suite2p_ds_annotated.path_full.iterdir():
            if not subdir.name.startswith("plane"):
                continue
            planei = int(subdir.name[5:])
            if empty_planes[planei]:
                continue
            if not subdir.is_dir():
                continue
            ops = np.load(subdir / "ops.npy", allow_pickle=True).item()
            # now we want to reextract the ROIs, so we set roidetect to True
            # It will not redo the detection if a stat.npy file is found
            ops["roidetect"] = True
            if ops["do_registration"] > 1:
                ops["do_registration"] = 1
            np.save(subdir / "ops.npy", ops, allow_pickle=True)

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
        suite2p_ds_annotated.path_full / "stat.npy",
        np.concatenate(all_stat, axis=0),
        allow_pickle=True,
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

    # Finally add some plots to check everything went fine: the original masks and the
    # new masks
    plot_reextraction_sanity(suite2p_ds_annotated, suite2p_ds, masks)

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


def verify_mask(path2mask, verbose=True):
    """Verifies that a mask file has unique and continuous cell IDs.

    Also checks that each ID corresponds to exactly one connected component.

    Args:
        path2mask (str or Path): Path to the mask image file.
        verbose (bool): Whether to print verification details. Defaults to True.

    Returns:
        tuple: (bool, set) where the bool is True if the mask is valid (continuous IDs
            and each ID is a single component), False otherwise and the set contains the
            IDs that correspond to multiple cells.
    """
    if verbose:
        print(f"Verifying {Path(path2mask).name}...")
    mask = load_mask(path2mask)
    unique_vals = np.unique(mask)

    # Check if 0 is present
    has_background = 0 in unique_vals

    # Check if continuous
    max_val = unique_vals.max()
    is_continuous = np.array_equal(unique_vals, np.arange(len(unique_vals)))

    if verbose and not is_continuous:
        print(f"  IDs are NOT continuous.")

    # Also check that each id corresponds to only 1 cell
    is_unique = True
    bad_masks = set()
    for mask_id in unique_vals:
        if mask_id == 0:
            continue
        binary_mask = mask == mask_id
        labeled_components = label(binary_mask, connectivity=2)
        n_comp = labeled_components.max()
        if n_comp > 1:
            is_unique = False
            bad_masks.add(mask_id)

    if verbose and not is_unique:
        print(f"  FAILURE: IDs {bad_masks} correspond to multiple cells.")

    return is_continuous and is_unique, bad_masks


def relabel_mask(mask_path, output_path=None, verbose=True):
    """Relabels a mask file to ensure unique and continuous cell IDs.

    Processes each original ID separately to split disconnected regions into unique
    new IDs while preserving separations between touching cells that already had
    different IDs.

    Args:
        mask_path (str or Path): Path to the input mask image file.
        output_path (str or Path, optional): Path where the relabeled mask will be
            saved. If None, the original file will be overwritten. Defaults to None.
        verbose (bool): Whether to print progress details. Defaults to True.
    """

    if verbose:
        print(f"Relabeling {Path(mask_path).name}...")
    mask = load_mask(mask_path)

    unique_ids = np.unique(mask)
    unique_ids = unique_ids[unique_ids != 0]

    new_mask = np.zeros_like(mask)
    next_id = 1

    for old_id in sorted(unique_ids):
        # Create a mask for just this ID
        id_mask = mask == old_id
        # Label components within this ID
        labeled_components = label(id_mask, connectivity=2)
        n_comp = labeled_components.max()

        for c in range(1, n_comp + 1):
            new_mask[labeled_components == c] = next_id
            next_id += 1
    if verbose:
        print(
            f"  Processed {len(unique_ids)} original IDs into {next_id - 1} unique components."
        )

    # Save as a new file
    if output_path is None:
        output_path = mask_path

    imsave(output_path, new_mask.astype(np.int16), check_contrast=False)
    if verbose:
        print(f"  Saved {Path(output_path).name}.")


def plot_reextraction_sanity(suite2p_ds, suite2p_ds_annotated, masks):
    """Plot the original masks and the new masks to check the reextraction.

    Args:
        suite2p_ds (Suite2pDataset): the original dataset
        suite2p_ds_annotated (Suite2pDataset): the annotated dataset
        masks (ndarray): the masks to be reextracted
    """

    # Load masks from the two suite2p_ds, iterate on the plane folders and read from ops

    fig, axes = plt.subplots(len(masks), 3, figsize=(10, len(masks) * 5), squeeze=False)

    for imask, mask in enumerate(masks):
        for col, ds in enumerate([suite2p_ds, suite2p_ds_annotated]):
            f, f_neu, spks, stats, iscell, ops = s2p_plotting_utils.load_s2p_output(
                ds.path_full / f"plane{imask}"
            )
            all_cells = s2p_plotting_utils.stats_to_array(
                stats, Ly=ops["Ly"], Lx=ops["Lx"], label_id=True
            )
            all_cells[all_cells == 0] = np.nan
            im = np.nanmax(all_cells, axis=0)
            axes[imask, col].imshow(im % 20, cmap="tab20", interpolation="none")
            if imask == 0:
                axes[imask, col].set_title(ds.dataset_name)
        mask = mask.astype(float)
        mask[mask == 0] = np.nan
        axes[imask, 2].imshow(mask % 20, cmap="tab20", interpolation="none")
        if imask == 0:
            axes[imask, 2].set_title("Input mask")
        axes[imask, 0].set_ylabel(f"Plane {imask}")
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(suite2p_ds_annotated.path_full / "reextraction_sanity.png")
