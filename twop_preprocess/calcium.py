import numpy as np
import os
import datetime
import flexiznam as flz
from flexiznam.schema import Dataset
from twop_preprocess.neuropil.ast_model import ast_model
import itertools
from suite2p import run_s2p
from suite2p.extraction import dcnv
from sklearn import mixture
from twop_preprocess.utils import parse_si_metadata, load_ops
from functools import partial
from tqdm import tqdm
from tifffile import TiffFile
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from numba import njit, prange
import matplotlib.pyplot as plt
from twop_preprocess.plotting_utils import sanity_check_utils as sanity

print = partial(print, flush=True)


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
            raise ValueError(f"No suite2p dataset found for session {session_name}. Cannot overwrite.")
        elif len(suite2p_datasets) > 1:
            print(
                f"{len(suite2p_datasets)} suite2p datasets found for session {session_name}"
            )
            print("Overwriting the last one...")
            suite2p_dataset = suite2p_datasets[
                np.argmax([datetime.datetime.strptime(i.created,'%Y-%m-%d %H:%M:%S')
                            for i in suite2p_datasets])]
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
    opsEnd = run_s2p(ops=ops, db=db)
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
        
        
def extract_dff(suite2p_dataset, ops):
    """
    Correct offsets, detrend, calculate dF/F and deconvolve spikes for the whole session.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        ops (dict): dictionary of suite2p settings

    """
    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    offsets = []
    for datapath in suite2p_dataset.extra_attributes["data_path"]:
        datapath = os.path.join(flz.PARAMETERS["data_root"]["raw"],
                                *datapath.split(os.path.sep)[-4:]) # add the raw path from flexiznam config
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
        Fneu = np.load(dpath / "Fneu.npy")
        if ops["sanity_plots"]:
            os.makedirs(dpath / "sanity_plots", exist_ok=True)
            np.random.seed(0)
            random_rois = np.random.choice(F.shape[0], ops["plot_nrois"], replace=False)
            sanity.plot_raw_trace(F, random_rois, Fneu)
            plt.savefig(dpath / "sanity_plots/raw_trace.png")
        F = correct_offset(dpath / "F.npy", offsets, first_frames[:, iplane], last_frames[:, iplane])
        Fneu = correct_offset(dpath / "Fneu.npy", offsets, first_frames[:, iplane], last_frames[:, iplane])  
        if ops["sanity_plots"]:
            sanity.plot_raw_trace(F, random_rois, Fneu)
            plt.savefig(dpath / "sanity_plots/offset_corrected.png")
            
        if ops["detrend"]:
            print("Detrending...")
            F_offset_corrected = F.copy()
            Fneu_offset_corrected = Fneu.copy()
            F, F_trend = detrend(F, first_frames[:, iplane], last_frames[:, iplane], ops, fs)
            Fneu, Fneu_trend = detrend(Fneu, first_frames[:, iplane], last_frames[:, iplane], ops, fs)     
            if ops["sanity_plots"]:
                sanity.plot_detrended_trace(F_offset_corrected, 
                                            F_trend, 
                                            F, 
                                            Fneu_offset_corrected, 
                                            Fneu_trend, 
                                            Fneu, 
                                            random_rois)
                plt.savefig(dpath / "sanity_plots/detrended.png")
                
        if ops["ast_neuropil"]:
            print("Running ASt neuropil correction...")
            correct_neuropil(dpath, F, Fneu)
            Fast = np.load(dpath / "Fast.npy")
            if ops["sanity_plots"]:
                sanity.plot_raw_trace(F, random_rois, Fast, titles=["F","Fast"])
                plt.savefig(dpath / "sanity_plots/neuropil_corrected.png")
                
        print("Calculating dF/F...")
        if ops["ast_neuropil"]:
            calculate_dFF(dpath, Fast, Fneu, ops)
        else:
            calculate_dFF(dpath, F, Fneu, ops)
        dff = np.load(dpath / "dff_ast.npy" if ops["ast_neuropil"] else dpath / "dff.npy")
        if ops["sanity_plots"]:
            F0 = np.load(dpath / "f0_ast.npy" if ops["ast_neuropil"] else dpath / "f0.npy")
            sanity.plot_dff(Fast, dff, F0, random_rois)
            plt.savefig(dpath / f'sanity_plots/dffs_n{ops["dff_ncomponents"]}.png')
            
        if ops["ast_neuropil"]:
            print("Deconvolve spikes from neuropil corrected trace...")
            spike_deconvolution_suite2p(suite2p_dataset, iplane, ops)  
              


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
    all_rec_baseline = np.zeros_like(F)
    for i, (start, end) in enumerate(zip(first_frames, last_frames)):
        rec_rolling_baseline  = np.zeros_like(F[:, start:end])
        for j in range(F.shape[0]):
            pad_front = win_frames // 2
            pad_end = win_frames // 2 - 1

            if ops["nplanes"] == 1:
                pad_end += 1
    
            rolling_baseline = np.pad(
                rolling_percentile(
                    F[j, start:end], 
                    win_frames,
                    ops["detrend_pctl"],
                ),
                (pad_front, pad_end),
                mode='edge',
            )

            rec_rolling_baseline[j, :] = rolling_baseline

        if i == 0:
            first_recording_baseline = np.median(rec_rolling_baseline, axis = 1)
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
    if not ops["ast_neuropil"]:
        F = F - ops["neucoeff"] * Fneu
    # Calculate dFFs and save to the suite2p folder
    print(f"n components for dFF calculation: {ops['dff_ncomponents']}")
    dff, f0 = dFF(F, n_components=ops["dff_ncomponents"])
    np.save(
        dpath / "dff_ast.npy" if ops["ast_neuropil"] else dpath / "dff.npy", dff
    )
    np.save(dpath / "f0_ast.npy" if ops["ast_neuropil"] else dpath / "f0.npy", f0)


def spike_deconvolution_suite2p(suite2p_dataset, iplane, ops={}):
    """
    Run spike deconvolution on the concatenated recordings after ASt neuropil correction.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        iplane (int): which plane to run on
        ops (dict): dictionary of suite2p settings

    """
    # Load the Fast.npy file and ops.npy file
    Fast_path = suite2p_dataset.path_full / f"plane{iplane}" / "Fast.npy"
    suite2p_ops_path = suite2p_dataset.path_full / f"plane{iplane}" / "ops.npy"
    Fast = np.load(Fast_path)
    suite2p_ops = np.load(suite2p_ops_path, allow_pickle=True).tolist()
    suite2p_ops.update(ops)
    ops = suite2p_ops

    # baseline operation
    Fast = dcnv.preprocess(
        F=Fast,
        baseline=ops["baseline_method"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
    )

    # get spikes
    spks_ast = dcnv.oasis(
        F=Fast, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"]
    )
    spks_ast_path = suite2p_dataset.path_full / f"plane{iplane}" / "spks_ast.npy"
    np.save(spks_ast_path, spks_ast)


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
        
    
def split_recordings(flz_session, suite2p_dataset, conflicts):
    """
    suite2p concatenates all the recordings in a given session into a single file.
    To facilitate downstream analyses, we cut them back into chunks and add them
    to flexilims as children of the corresponding recordings.

    Args:
        flz_session (Flexilims): flexilims session
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        conflicts (str): defines behavior if recordings have already been split

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
    datasets_out = []
    first_frames, last_frames = get_recording_frames(suite2p_dataset)
    nplanes = int(float(suite2p_dataset.extra_attributes["nplanes"]))

    split_datasets = []
    for raw_datapath, recording_id, first_frames_rec, last_frames_rec in zip(
        datapaths, recording_ids, first_frames, last_frames
    ):
        # minimum number of frames across planes
        nframes = np.min(last_frames_rec - first_frames_rec)
        # get the path for split dataset
        split_dataset = flz.get_datasets(
            origin_id=recording_id,
            dataset_type="suite2p_traces",
            project_id=suite2p_dataset.project,
            flexilims_session=flz_session,
            return_dataseries=False,
        )
        
        recording_name = flz.get_entity(datatype="recording",flexilims_session=flz_session,id=recording_id).name
        if len(split_dataset) > 0:
            print(
                f"WARNING:{len(split_dataset)} suite2p datasets found for recording {recording_name}"
            )
            split_dataset = split_dataset[
                np.argmax([datetime.datetime.strptime(i.created,'%Y-%m-%d %H:%M:%S')
                            for i in split_dataset])]
            print(split_dataset)
            if conflicts == "overwrite":
                print(f"Overwriting the last dataset {split_dataset.full_name}...")
            elif (split_dataset.get_flexilims_entry() is not None) and (conflicts == "skip"):
                print(f"Dataset {split_dataset.full_name} already split... skipping...")
                datasets_out.append(split_dataset)
                continue  
            
            else: 
                split_dataset = Dataset.from_origin(
                    project=suite2p_dataset.project,
                    origin_type="recording",
                    origin_id=recording_id,
                    dataset_type="suite2p_traces",
                    conflicts=conflicts,
                )
        else:
            split_dataset = Dataset.from_origin(
                project=suite2p_dataset.project,
                origin_type="recording",
                origin_id=recording_id,
                dataset_type="suite2p_traces",
                conflicts=conflicts,
            )
            
        split_dataset.path_full.mkdir(parents=True, exist_ok=True)
        si_metadata = parse_si_metadata(raw_datapath)
        np.save(split_dataset.path_full / "si_metadata.npy", si_metadata)
        # load processed data
        for iplane, start in zip(range(nplanes), first_frames_rec):
            suite2p_path = suite2p_dataset.path_full / f"plane{iplane}"
            # otherwise lets split it
            split_path = split_dataset.path_full / f"plane{iplane}"
            try:
                split_path.mkdir(parents=True, exist_ok=True)
            except OSError:
                print(f"Error creating directory {split_path}")
            F, Fneu, spks = (
                np.load(suite2p_path / "F.npy"),
                np.load(suite2p_path / "Fneu.npy"),
                np.load(suite2p_path / "spks.npy"),
            )
            if suite2p_dataset.extra_attributes["ast_neuropil"]:
                Fast, dff_ast, spks_ast = (
                    np.load(suite2p_path / "Fast.npy"),
                    np.load(suite2p_path / "dff_ast.npy"),
                    np.load(suite2p_path / "spks_ast.npy"),
                )
            end = start + nframes
            np.save(split_path / "F.npy", F[:, start:end])
            np.save(split_path / "Fneu.npy", Fneu[:, start:end])
            np.save(split_path / "spks.npy", spks[:, start:end])
            if suite2p_dataset.extra_attributes["ast_neuropil"]:
                np.save(split_path / "Fast.npy", Fast[:, start:end])
                np.save(split_path / "dff_ast.npy", dff_ast[:, start:end])
                np.save(split_path / "spks_ast.npy", spks_ast[:, start:end])
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
                np.argmax([datetime.datetime.strptime(i.created,'%Y-%m-%d %H:%M:%S')
                            for i in suite2p_datasets])]
        else:
            suite2p_dataset = suite2p_datasets[0]

    if run_dff:
        print("Calculating dF/F...")
        extract_dff(suite2p_dataset, ops)

    if run_split:
        print("Splitting recordings...")
        split_recordings(flz_session, suite2p_dataset, conflicts=conflicts)
