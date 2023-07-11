import numpy as np
import datetime
import flexiznam as flz
from flexiznam.schema import Dataset
from twop_preprocess.neuropil import correct_neuropil
import itertools
from suite2p import run_s2p
from suite2p.extraction import dcnv
from sklearn import mixture
from twop_preprocess.utils import parse_si_metadata, load_ops
from functools import partial
from tqdm import tqdm

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
    si_datasets = flz.get_datasets(
        exp_session["id"],
        recording_type="two_photon",  #!Some 2p recording may not be labelled as 'two_photon'
        dataset_type="scanimage",
        flexilims_session=flz_session,
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
    ops["nplanes"] = si_metadata["SI.hStackManager.numSlices"]
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
    # run neuropil correction, dFF calculation and spike deconvolution
    for iplane in range(opsEnd["nplanes"]):
        if ops["ast_neuropil"]:
            print("Running ASt neuropil correction...")
            correct_neuropil(suite2p_dataset, iplane)
        print("Calculating dF/F...")
        calculate_dFF(suite2p_dataset, iplane, ops)
        if ops["ast_neuropil"]:
            print("Deconvolve spikes from neuropil corrected trace...")
            spike_deconvolution_suite2p(suite2p_dataset, iplane)

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
    return suite2p_dataset, opsEnd


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


def calculate_dFF(suite2p_dataset, iplane, ops):
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
    # Load fluorescence traces
    dir_path = suite2p_dataset.path_full / f"plane{iplane}"
    if ops["ast_neuropil"]:
        F = np.load(dir_path / "Fast.npy")
    else:
        F = np.load(dir_path / "F.npy")
        Fneu = np.load(dir_path / "Fneu.npy")
        F = F - ops["neucoeff"] * Fneu
    # Calculate dFFs and save to the suite2p folder
    dff, f0 = dFF(F, n_components=ops["dff_ncomponents"])
    np.save(
        dir_path / "dff_ast.npy" if ops["ast_neuropil"] else dir_path / "dff.npy", dff
    )
    np.save(dir_path / "f0_ast.npy" if ops["ast_neuropil"] else dir_path / "f0.npy", f0)


def spike_deconvolution_suite2p(suite2p_dataset, iplane):
    """
    Run spike deconvolution on the concatenated recordings after ASt neuropil correction.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
        iplane (int): which plane to run on
        baseline (str): method for baseline estimation before spike deconvolution. Default 'maximin'.
        sig_baseline (float): standard deviation of gaussian with which to smooth. Default 10.0.
        win_baseline (float): window in which to compute max/min filters in seconds. Default 60.0.

    """
    # Load the Fast.npy file and ops.npy file
    Fast_path = suite2p_dataset.path_full / f"plane{iplane}" / "Fast.npy"
    ops_path = suite2p_dataset.path_full / f"plane{iplane}" / "ops.npy"
    Fast = np.load(Fast_path)
    ops = np.load(ops_path, allow_pickle=True).tolist()

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
    # load the ops file to find length of individual recordings
    ops_path = suite2p_dataset.path_full / "plane0" / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).item()
    # get scanimage datasets
    datasets = flz.get_datasets(
        suite2p_dataset.origin_id,
        recording_type="two_photon",
        dataset_type="scanimage",
        flexilims_session=flz_session,
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
    # split into individual recordings
    assert len(datapaths) == len(ops["frames_per_folder"])
    last_frames = np.cumsum(ops["frames_per_folder"])
    first_frames = np.concatenate(([0], last_frames[:-1]))
    datasets_out = []
    for raw_datapath, recording_id, start, end in zip(
        datapaths, recording_ids, first_frames, last_frames
    ):
        split_dataset = Dataset.from_origin(
            project=suite2p_dataset.project,
            origin_type="recording",
            origin_id=recording_id,
            dataset_type="suite2p_traces",
            conflicts=conflicts,
        )
        if (split_dataset.get_flexilims_entry() is not None) and conflicts == "skip":
            print(f"Dataset {split_dataset.name} already split... skipping...")
            datasets_out.append(split_dataset)
            continue
        split_dataset.path_full.mkdir(parents=True, exist_ok=True)
        si_metadata = parse_si_metadata(raw_datapath)
        np.save(split_dataset.path_full / "si_metadata.npy", si_metadata)
        # load processed data
        for iplane in range(ops["nplanes"]):
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
        split_dataset.update_flexilims(mode="overwrite")
        datasets_out.append(split_dataset)
    return datasets_out


def extract_session(
    project,
    session_name,
    conflicts=None,
    run_split=False,
    run_extraction=True,
    ops={},
):
    """
    Process all the 2p datasets for a given session

    Args:
        project (str): name of the project, e.g. '3d_vision'
        session_name (str): name of the session
        conflicts (str): how to treat existing processed data
        run_split (bool): whether or not to run splitting for different folders
        run_extraction (bool): whether or not to run extraction

    """
    # get session info from flexilims
    print("Connecting to flexilims...")
    flz_session = flz.get_flexilims_session(project)
    ops = load_ops(ops)
    if run_extraction:
        suite2p_dataset, opsEnd = run_extraction(
            flz_session, project, session_name, conflicts, ops
        )
    else:
        session_children = flz.get_children(
            parent_name=session_name,
            children_datatype="dataset",
            flexilims_session=flz_session,
        )
        suite2p_datasets = session_children[
            session_children["dataset_type"] == "suite2p_rois"
        ]
        if len(suite2p_datasets) == 0:
            raise ValueError(f"No suite2p dataset found for session {session_name}")
        elif len(suite2p_datasets) > 1:
            print(
                f"{len(suite2p_datasets)} suite2p datasets found for session {session_name}"
            )
            print("Splitting the last one...")
        suite2p_dataset = Dataset.from_dataseries(
            suite2p_datasets.iloc[-1], flexilims_session=flz_session
        )

    if run_split:
        print("Splitting recordings...")
        split_recordings(flz_session, suite2p_dataset, conflicts=conflicts)
