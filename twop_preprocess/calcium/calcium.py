import numpy as np
import os
import datetime
import flexiznam as flz
import itertools
from znamutils import slurm_it
from functools import partial
import matplotlib.pyplot as plt

from ..utils import parse_si_metadata, load_ops
from ..plotting_utils import sanity_check_utils as sanity
from .processing_steps import estimate_offsets, detrend
from .calcium_s2p import run_extraction, spike_deconvolution_suite2p
from .calcium_utils import (
    calculate_and_save_dFF,
    correct_neuropil_ast,
    correct_neuropil_standard,
    correct_offset,
    get_recording_frames,
)


print = partial(print, flush=True)


def process_concatenated_traces(suite2p_dataset, ops, project, flz_session):
    """
    Correct offsets, detrend, calculate dF/F and deconvolve spikes for the whole session.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        ops (dict): dictionary of suite2p settings

    """
    print("Starting processing of concatenated traces...")
    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    fs = suite2p_dataset.extra_attributes["fs"]
    nplanes = int(suite2p_dataset.extra_attributes["nplanes"])

    # --- 1. Estimate Offsets (once per session) ---
    offsets = estimate_offsets(suite2p_dataset, ops, project, flz_session)

    # --- 2. Process each plane ---
    # run neuropil correction, dFF calculation and spike deconvolution
    for iplane in range(nplanes):
        print(f"\n--- Processing Plane {iplane} ---")
        dpath = suite2p_dataset.path_full / f"plane{iplane}"
        plot_path = dpath / "sanity_plots"

        # --- 2a. Load Data ---
        try:
            F_raw = np.load(dpath / "F.npy")
            Fneu_raw = np.load(dpath / "Fneu.npy")
            if F_raw.shape[1] == 0:
                print(f"No ROIs found for plane {iplane}. Skipping plane.")
                continue
            if F_raw.shape != Fneu_raw.shape:
                raise ValueError(f"F and Fneu shapes mismatch for plane {iplane}")
        except FileNotFoundError:
            print(f"F.npy or Fneu.npy not found for plane {iplane}. Skipping plane.")
            continue
        except ValueError as e:
            print(f"Error loading data for plane {iplane}: {e}. Skipping plane.")
            continue

        # Prepare for plotting if needed
        random_rois = None
        do_plots = ops.get("sanity_plots", False)
        if do_plots:
            plot_path.mkdir(exist_ok=True)
            np.random.seed(0)
            n_rois_to_plot = min(ops.get("plot_nrois", 5), F_raw.shape[0])
            if n_rois_to_plot > 0:
                random_rois = np.random.choice(
                    F_raw.shape[0], n_rois_to_plot, replace=False
                )
            else:
                print(
                    "Warning: plot_nrois is 0 or invalid, cannot generate ROI-specific plots."
                )
                random_rois = np.array([])  # Ensure it's an array for consistency
            do_plots = random_rois is not None and len(random_rois) > 0
            if do_plots:
                print(
                    f"Generating sanity plots for {len(random_rois)} random ROIs: {random_rois}"
                )
                sanity.plot_raw_trace(
                    F_raw,
                    random_rois,
                    Fneu_raw,
                    save_path=plot_path / "01_raw_traces.png",
                )

        # --- 2b. Correct Offset ---
        if ops.get("correct_offset", True):
            print("Correcting offset...")
            F_offset_corrected = correct_offset(
                dpath / "F.npy",
                offsets,
                first_frames[:, iplane],
                last_frames[:, iplane],
            )
            Fneu_offset_corrected = correct_offset(
                dpath / "Fneu.npy",
                offsets,
                first_frames[:, iplane],
                last_frames[:, iplane],
            )
        else:
            print("Offset correction skipped.")
            F_offset_corrected = F_raw
            Fneu_offset_corrected = Fneu_raw
        if do_plots:
            sanity.plot_raw_trace(
                F_offset_corrected,
                random_rois,
                Fneu_offset_corrected,
                save_path=plot_path / "02_offset_corrected.png",
            )

        # --- 2c. Detrend ---
        if ops.get("detrend", True):
            print("Detrending...")
            F_detrended, F_trend = detrend(
                F_offset_corrected,
                first_frames[:, iplane],
                last_frames[:, iplane],
                ops,
                fs,
            )
            Fneu_detrended, Fneu_trend = detrend(
                Fneu_offset_corrected,
                first_frames[:, iplane],
                last_frames[:, iplane],
                ops,
                fs,
            )
            if do_plots:
                sanity.plot_detrended_trace(
                    F_offset_corrected,
                    F_trend,
                    F_detrended,
                    Fneu_offset_corrected,
                    Fneu_trend,
                    Fneu_detrended,
                    random_rois,
                    save_path=plot_path / "03_detrended.png",
                )
        else:
            print("Detrending skipped.")
            F_detrended = F_offset_corrected
            Fneu_detrended = Fneu_offset_corrected

        # --- 2d. Neuropil Correction ---
        F_processed = None
        if ops["ast_neuropil"]:
            print("Running ASt neuropil correction...")
            Fast = correct_neuropil_ast(dpath, F_detrended, Fneu_detrended)
            F_processed = Fast
            filename_suffix = "_ast"
        else:
            neucoef = ops.get("neucoeff", 0.7)
            print(f"Running standard neuropil correction with coefficient {neucoef}...")
            Fstandard = correct_neuropil_standard(
                F_detrended, Fneu_detrended, neucoef, save_path=dpath / "Fstandard.npy"
            )
            F_processed = Fstandard
            filename_suffix = ""

        if do_plots:
            sfx = "AST" if ops["ast_neuropil"] else "Standard"
            sanity.plot_neuropil_corrected_trace(
                F_detrended,
                random_rois,
                F_processed,
                titles=["F Detrended", f"F Corrected ({sfx})"],
                save_path=plot_path / "04b_neuropil_corrected.png",
            )

        # --- 2e. Calculate dF/F ---
        print("Calculating dF/F")
        dff, f0 = calculate_and_save_dFF(
            dpath, F_processed, filename_suffix, ops.get("dff_ncomponents", 2)
        )
        if do_plots:
            sanity.plot_dff(
                F_processed,
                dff,
                f0,
                random_rois,
                save_path=plot_path / f"05_dff{filename_suffix}.png",
            )
            fig = sanity.plot_fluorescence_matrices(
                F_detrended, Fneu_detrended, Fast, dff, ops["neucoeff"]
            )
            fig.savefig(plot_path / f"fluorescence_matrices.png")
            # Plot GMM for baseline estimation (using F_proc)
            for roi in random_rois:
                sanity.plot_offset_gmm(
                    F_processed,
                    roi,
                    ops.get("dff_ncomponents", 2),
                    save_path=plot_path / f"07_dff_gmm_roi{roi}{filename_suffix}.png",
                )

        # --- 2f. Spike Deconvolution ---
        print("Deconvolve spikes ...")
        spike_deconvolution_suite2p(
            suite2p_dataset, iplane, ops, ast_neuropil=ops["ast_neuropil"]
        )
        print(f"--- Finished processing Plane {iplane} ---")
    print("\nFinished processing all planes for concatenated traces.")


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
        split_dataset.update_flexilims(mode="overwrite")
        datasets_out.append(split_dataset)
    return datasets_out


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
def extract_session(
    project,
    session_name,
    conflicts=None,
    run_split=False,
    run_suite2p=True,
    run_dff=True,
    delete_previous_run=False,
    ops=None,
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
        delete_previous_run (bool): whether to delete previous runs. Default False
        ops (dict): dictionary of suite2p settings
        use_slurm (bool): whether to use slurm or not. Default False
        slurm_folder (Path): path to the slurm folder. Default None
        scripts_name (str): name of the slurm script. Default None

    Returns:
        Dataset: object containing the generated dataset


    """
    if ops is None:
        ops = {}

    # get session info from flexilims
    print("Connecting to flexilims...")
    flz_session = flz.get_flexilims_session(project)
    ops = load_ops(ops)
    if run_suite2p:
        suite2p_dataset = run_extraction(
            flz_session, project, session_name, conflicts, ops, delete_previous_run
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
        process_concatenated_traces(suite2p_dataset, ops, project, flz_session)

    if run_split:
        print("Splitting recordings...")
        split_recordings(flz_session, suite2p_dataset, conflicts=conflicts)
    print("Extraction finished.")
