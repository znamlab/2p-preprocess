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

    This function iterates through all planes of a Suite2p dataset, applying a series
    of preprocessing steps (offset correction, detrending, neuropil correction,
    dF/F calculation, and spike deconvolution). It also generates sanity plots if
    configured in the ops.

    Args:
        suite2p_dataset (Dataset): Flexilims Dataset object containing concatenated recordings.
        ops (dict): Dictionary of preprocessing settings (e.g., 'correct_offset', 'detrend',
            'ast_neuropil', 'sanity_plots').
        project (str): Flexilims project name.
        flz_session (Flexilims): Active Flexilims session object.
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
                valid_rois = np.where(~np.isnan(F_raw).all(axis=1))[0]
                if len(valid_rois) == 0:
                    raise ValueError("F for all rois is NaN")

                random_rois = np.random.choice(
                    valid_rois, n_rois_to_plot, replace=False
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
            Fast = np.zeros_like(F_detrended)

        if do_plots:
            sfx = "AST" if ops["ast_neuropil"] else "Standard"
            sanity.plot_raw_trace(
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
                    F_detrended,
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
            on flexilims. Used to filter when looking for previously split datasets.
            Default None

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
            extra_attributes=extra_attributes,
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
            do_ast = suite2p_dataset.extra_attributes["ast_neuropil"]
            if extra_attributes is not None:
                if "ast_neuropil" in extra_attributes:
                    do_ast = extra_attributes["ast_neuropil"]
            if do_ast and (do_ast != "False"):
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
    Main entry point to process all 2p datasets for a given session.

    This function coordinates the full pipeline: running Suite2p extraction
    (or loading existing results), calculating dF/F with optional neuropil
    correction, and optionally splitting the concatenated traces back into
    individual recording datasets on Flexilims.

    Args:
        project (str): Name of the Flexilims project (e.g., '3d_vision').
        session_name (str): Name of the experimental session.
        conflicts (str, optional): How to handle existing data on Flexilims
            ('skip', 'overwrite', 'abort').
        run_split (bool, optional): Whether to split concatenated traces into
            individual recording datasets. Default False.
        run_suite2p (bool, optional): Whether to run Suite2p ROI extraction. Default True.
        run_dff (bool, optional): Whether to run dF/F calculation and neuropil correction.
            Default True.
        delete_previous_run (bool, optional): If True, deletes existing Suite2p binary files
            and output before starting. Default False.
        ops (dict, optional): Dictionary of Suite2p and preprocessing settings.
            If None, default ops are loaded.

    Returns:
        Dataset: The Flexilims Dataset object for the Suite2p ROIs.
    """
    if ops is None:
        ops = {}
    print("Running extract_session with parameters:")
    print(f"Project: {project}")
    print(f"Session: {session_name}")
    print(f"Conflicts: {conflicts}")
    print(f"Run split: {run_split}")
    print(f"Run suite2p: {run_suite2p}")
    print(f"Run dff: {run_dff}")
    print(f"Delete previous run: {delete_previous_run}")
    print(f"Ops: {ops}")
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
        split_recordings(
            flz_session,
            suite2p_dataset,
            conflicts=conflicts,
            extra_attributes={"ast_neuropil": ops["ast_neuropil"]},
        )
    print("Extraction finished.")


def generate_sanity_plots(project, session_name, flz_session):
    """
    Re-generate all sanity plots for a previously processed session.

    This function loads the Suite2p dataset, re-calculates necessary intermediate
    steps (like detrending) that aren't saved to disk, and recreates the full
    suite of diagnostic figures.

    Args:
        project (str): Flexilims project name.
        session_name (str): Name of the experimental session.
        flz_session (Flexilims): Active Flexilims session object.
    """
    print(f"Generating sanity plots for session {session_name}...")

    # get experimental session
    exp_session = flz.get_entity(
        datatype="session",
        name=session_name,
        project_id=project,
        flexilims_session=flz_session,
    )
    if exp_session is None:
        raise ValueError(f"Session {session_name} not found on flexilims")

    # fetch existing suite2p dataset
    suite2p_datasets = flz.get_datasets(
        origin_id=exp_session["id"],
        dataset_type="suite2p_rois",
        project_id=project,
        flexilims_session=flz_session,
    )
    if not suite2p_datasets:
        raise ValueError(f"No Suite2p dataset found for session {session_name}")

    # Use the most recent one if multiple exist
    if len(suite2p_datasets) > 1:
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

    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    fs = suite2p_dataset.extra_attributes["fs"]
    nplanes = int(suite2p_dataset.extra_attributes["nplanes"])

    # --- 1. Re-estimate Offsets (generates optical offset plot) ---
    # We pass sanity_plots=True implicitly to ensure the plot is generated
    ops_base = suite2p_dataset.extra_attributes.copy()
    ops_base["sanity_plots"] = True
    offsets = estimate_offsets(suite2p_dataset, ops_base, project, flz_session)

    # --- 2. Process each plane ---
    for iplane in range(nplanes):
        print(f"\n--- Plotting Plane {iplane} ---")
        dpath = suite2p_dataset.path_full / f"plane{iplane}"
        plot_path = dpath / "sanity_plots"
        plot_path.mkdir(exist_ok=True)

        # Load plane-specific ops
        ops = np.load(dpath / "ops.npy", allow_pickle=True).item()
        ops["sanity_plots"] = True

        F_raw = np.load(dpath / "F.npy")
        Fneu_raw = np.load(dpath / "Fneu.npy")

        # Select ROIs for plotting
        np.random.seed(0)
        n_rois_to_plot = min(ops.get("plot_nrois", 5), F_raw.shape[0])
        valid_rois = np.where(~np.isnan(F_raw).all(axis=1))[0]
        if len(valid_rois) == 0:
            print(f"No valid ROIs for plane {iplane}. Skipping.")
            continue
        random_rois = np.random.choice(valid_rois, n_rois_to_plot, replace=False)

        # 01. Raw
        sanity.plot_raw_trace(
            F_raw,
            random_rois,
            Fneu_raw,
            save_path=plot_path / "01_raw_traces.png",
        )

        # 02. Offset Corrected
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
        sanity.plot_raw_trace(
            F_offset_corrected,
            random_rois,
            Fneu_offset_corrected,
            save_path=plot_path / "02_offset_corrected.png",
        )

        # 03. Detrended
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

        # 04. Neuropil Corrected
        ast_enabled = ops.get("ast_neuropil", True)
        processed_file = "Fast.npy" if ast_enabled else "Fstandard.npy"
        filename_suffix = "_ast" if ast_enabled else ""
        sfx = "AST" if ast_enabled else "Standard"

        if (dpath / processed_file).exists():
            F_processed = np.load(dpath / processed_file)
            sanity.plot_raw_trace(
                F_detrended,
                random_rois,
                F_processed,
                titles=["F Detrended", f"F Corrected ({sfx})"],
                save_path=plot_path / "04b_neuropil_corrected.png",
            )
        else:
            print(
                f"Warning: {processed_file} not found. Skipping neuropil correction plots."
            )
            F_processed = None
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"{processed_file} not found.\nNeuropil correction plot skipped.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            fig.savefig(plot_path / "04b_neuropil_corrected.png")
            plt.close(fig)

        # 05. dF/F
        dff_file = dpath / f"dff{filename_suffix}.npy"
        f0_file = dpath / f"f0{filename_suffix}.npy"

        if dff_file.exists() and f0_file.exists() and F_processed is not None:
            dff = np.load(dff_file)
            f0 = np.load(f0_file)
            sanity.plot_dff(
                F_processed,
                dff,
                f0,
                random_rois,
                save_path=plot_path / f"05_dff{filename_suffix}.png",
            )

            # 06. Matrix heatmaps
            Fast = F_processed if ast_enabled else np.zeros_like(F_detrended)
            fig = sanity.plot_fluorescence_matrices(
                F_detrended, Fneu_detrended, Fast, dff, ops.get("neucoeff", 0.7)
            )
            fig.savefig(plot_path / f"fluorescence_matrices.png")
            plt.close(fig)

            # 07. GMM f0 fits
            for roi in random_rois:
                sanity.plot_offset_gmm(
                    F_detrended,
                    F_processed,
                    roi,
                    ops.get("dff_ncomponents", 2),
                    save_path=plot_path / f"07_dff_gmm_roi{roi}{filename_suffix}.png",
                )
                plt.close()
        else:
            print(f"Warning: dF/F files or F_processed missing. Skipping dF/F plots.")
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                f"dF/F files or corrected trace missing.\ndF/F plots skipped.",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            fig.savefig(plot_path / f"05_dff{filename_suffix}.png")
            plt.close(fig)

    print(f"\nFinished generating sanity plots for session {session_name}.")
