from functools import partial
import os

import matplotlib.pyplot as plt
import flexiznam as flz
import numpy as np
from znamutils import slurm_it
import datetime

from ..plotting_utils import sanity_check_utils as sanity
from ..utils import load_ops
from .postprocessing import split_recordings
from .suite2p_interaction import run_extraction, spike_deconvolution_suite2p
from .calcium_processing import (
    estimate_offset,
    correct_offset,
    detrend,
    calculate_dFF,
    correct_neuropil,
)

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
        extract_dff(suite2p_dataset, ops, project, flz_session)

    if run_split:
        print("Splitting recordings...")
        split_recordings(flz_session, suite2p_dataset, conflicts=conflicts)
    print("Extraction finished.")


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
        datapath = os.path.join(
            flz.get_data_root("raw", project, flz_session), *datapath.split("/")[-4:]
        )  # add the raw path from flexiznam config
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

            else:
                # Plot random cells
                for roi in random_rois:
                    sanity.plot_offset_gmm(F, Fneu, roi, ops["dff_ncomponents"])
                    plt.savefig(dpath / "sanity_plots" / f"offset_gmm_roi{roi}.png")


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
    try:
        nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))

    except KeyError:  # Default to 1 if missing
        suite2p_dataset.extra_attributes["nplanes"] = 1
        suite2p_dataset.update_flexilims(mode="update")

        nplanes = int(suite2p_dataset.extra_attributes["nplanes"])

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
