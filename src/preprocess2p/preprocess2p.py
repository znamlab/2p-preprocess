import numpy as np
import defopt
import os
import datetime
import flexiznam as flz
from flexiznam.schema import Dataset
from pathlib import Path
from neuropil import correct_neuropil
import itertools

from suite2p import run_s2p, default_ops
from suite2p.extraction import dcnv
from tifffile import TiffFile, TiffWriter
from more_itertools import chunked
import scipy.fft as fft
from sklearn import mixture
from tqdm import tqdm


def phase_corr(
    reference: np.ndarray,
    target: np.ndarray,
    max_shift=None,
    whiten=True,
    fft_ref=True,
) -> np.ndarray:
    """
    Compute phase correlation of two images.

    Args:
        reference (numpy.ndarray): reference image
        target (numpy.ndarray): target image
        max_shift (int): the range over which to search for the maximum of the
            cross-correlogram
        whiten (bool): whether or not to whiten the FFTs of the images. If True,
            the method performs phase correlation, otherwise cross correlation
        fft_ref (bool): whether to compute the FFT transform of the reference

    Returns:
        shift: numpy.array of the location of the peak of the cross-correlogram
        cc: numpy.ndarray of the cross-correlagram itself.

    """
    if fft_ref:
        f1 = fft.fft2(reference)
    else:
        f1 = reference
    f2 = fft.fft2(target)
    if whiten:
        f1 = f1 / np.abs(f1)
        f2 = f2 / np.abs(f2)
    cc = np.abs(fft.ifft2(f1 * np.conj(f2)))
    if max_shift:
        cc[max_shift:-max_shift, :] = 0
        cc[:, max_shift:-max_shift] = 0
    cc = fft.fftshift(cc)

    shift = (
        np.unravel_index(np.argmax(cc), reference.shape) - np.array(reference.shape) / 2
    )
    return shift, cc


def parse_si_metadata(tiff_path):
    """
    Reads metadata from a Scanimage TIFF and return a dictionary with
    specified key values.

    Currently can only extract numerical data.

    Args:
        tiff_path: path to TIFF or directory containing tiffs

    Returns:
        dict: dictionary of SI parameters

    """
    if not tiff_path.endswith(".tif"):
        tiffs = [tiff for tiff in os.listdir(tiff_path) if tiff.endswith(".tif")]
    else:
        tiffs = [
            tiff_path,
        ]
    if tiffs:
        tiff_path = str(Path(tiff_path) / tiffs[0])
        tif = TiffFile(tiff_path)
        return tif.scanimage_metadata["FrameData"]
    else:
        return None


def estimate_bidi_correction(im):
    odd_rows = im[1::2, :]
    even_rows = im[::2, :]

    shift = phase_corr(even_rows, odd_rows, max_shift=20)[0][1]
    return shift


def apply_bidi_correction(im, shift):
    if im.ndim == 2:
        im = np.expand_dims(im, axis=0)
    odd_rows = im[:, 1::2, :]
    odd_rows = np.roll(odd_rows, int(shift), axis=2)
    im[:, 1::2, :] = odd_rows
    return np.squeeze(im)


def register_zstack(
    tiff_paths,
    ch_to_align=0,
    iter=1,
    max_shift=50,
    align_planes=True,
    bidi_correction=False,
):
    """
    Apply motion correction to a z-stack.

    We first apply motion correction to individual frames in each slice and the
    register adjacent slices to each other.

    Args:
        tiff_paths (list): list of full paths to the z-stack files
        ch_to_align (int): channel to use for registration
        nchannels (int): number of channels in the stack
        iter (int): number of iterations to perform for each plane to refine the
            registration template (default: 1)
        max_shift (int): maximum shift to search for in the cross-correlogram
            (default: 50)
        align_planes (bool): whether or not to align planes to each other
            (default: True)
        bidi_correction (bool): whether or not to apply bidirectional scanning
            correction. If True, the method will estimate the shift between
            odd and even rows and apply it to the odd rows.

    Rerurns:
        numpy.ndarray: z-stack after applying motion correction (X x Y x Z)
        nz: int, number of slices in aligned_stack
        nchannels: int, number of channels in aligned_stack

    """
    # get the aquisition params from the first .tif file
    si_dict = parse_si_metadata(tiff_paths[0])
    nchannels = len(si_dict["SI.hChannels.channelSave"])
    assert nchannels > ch_to_align
    # get a list of stacks from the acquisition
    stack_list = [TiffFile(tiff_path) for tiff_path in tiff_paths]
    # chain the the pages from the stack
    stack_pages = itertools.chain(*[stack.pages for stack in stack_list])
    nframes = int(si_dict["SI.hStackManager.framesPerSlice"])

    chunk_size = nframes * nchannels

    nx = int(si_dict["SI.hRoiManager.pixelsPerLine"])
    ny = int(si_dict["SI.hRoiManager.linesPerFrame"])
    nz = int(si_dict["SI.hStackManager.actualNumSlices"])

    registered_stack = np.zeros((nx, ny, nchannels, nz))
    frame_shifts = np.zeros((nframes, 2, nz))
    # process stack one slice at a time
    for iplane, plane in enumerate(chunked(stack_pages, chunk_size)):
        print(f"Registering plane {iplane+1} of {nz}", flush=True)
        data = np.asarray([page.asarray() for page in plane])
        if bidi_correction:
            if iplane == 0:
                bidi_shift = estimate_bidi_correction(data[ch_to_align, :, :])
                print(f"Estimated bidi shift: {bidi_shift}", flush=True)
            data = apply_bidi_correction(data, bidi_shift)
        # generate reference image for the current slice
        for i in range(iter):
            if i == 0:
                template_image = np.mean(data[ch_to_align::nchannels, :, :], axis=0)
            else:
                template_image = registered_stack[:, :, ch_to_align, iplane]
                registered_stack[:, :, ich, iplane] = 0
            template_image_fft = fft.fft2(template_image)
            # use reference image to align individual planes
            for iframe in tqdm(range(nframes)):
                shifts = phase_corr(
                    template_image_fft,
                    data[nchannels * iframe + ch_to_align, :, :],
                    max_shift=max_shift,
                    whiten=True,
                    fft_ref=False,
                )[0]
                frame_shifts[iframe, :, iplane] = shifts
                for ich in range(nchannels):
                    registered_stack[:, :, ich, iplane] += np.roll(
                        data[nchannels * iframe + ich, :, :],
                        (int(shifts[0]), int(shifts[1])),
                        axis=(0, 1),
                    )
    plane_shifts = np.zeros((2, nz))
    if not align_planes:
        return (
            registered_stack / int(nframes),
            nz,
            nchannels,
            frame_shifts,
            plane_shifts,
        )
    aligned_stack = np.zeros((nx, ny, nchannels, nz))
    # we don't need to align the very first plane
    aligned_stack[:, :, :, 0] = registered_stack[:, :, :, 0]

    # align planes to each other
    print("Aliging planes to each other", flush=True)
    for iplane in range(1, nz):
        shifts = phase_corr(
            aligned_stack[:, :, ch_to_align, iplane - 1],
            registered_stack[:, :, ch_to_align, iplane],
            max_shift=max_shift,
            whiten=True,
            fft_ref=True,
        )[0]
        plane_shifts[:, iplane] = shifts
        for ich in range(nchannels):
            aligned_stack[:, :, ich, iplane] = np.roll(
                registered_stack[:, :, ich, iplane],
                (int(shifts[0]), int(shifts[1])),
                axis=(0, 1),
            )
            if shifts[0] > 0:
                aligned_stack[: int(shifts[0]), :, ich, iplane] = 0
            else:
                aligned_stack[-int(shifts[0]) :, :, ich, iplane] = 0
            if shifts[1] > 0:
                aligned_stack[:, : int(shifts[1]), ich, iplane] = 0
            else:
                aligned_stack[:, -int(shifts[1]) :, ich, iplane] = 0

    return aligned_stack / int(nframes), nz, nchannels, frame_shifts, plane_shifts


def run_zstack_registration(
    flz_session, project, session_name, conflicts="append", ch_to_align=0
):
    """
    Apply motion correction to all zstacks for a single session, create Flexylims
    entries for each registered zstacks

    Args:
        flz_session ()
        project (str): human-readable string for project name in Flexylims
                (hexadecimal id fails with Dataset.from_origin)
        session_name (str): string matching Flexylims session name
        conflicts (str): string for handling flexilims conflicts, if more than
                one zstack needs to be registered for session, use conflicts="append"
        ch_to_align (int): channel to use for calculating shifts

    """
    # get experimental session
    exp_session = flz.get_entity(
        datatype="session", name=session_name, flexilims_session=flz_session
    )

    # get all zstacks from session
    zstacks = flz.get_entities(
        datatype="dataset",
        origin_id=exp_session["id"],
        query_key="stack_type",
        query_value="zstack",
        flexilims_session=flz_session,
    )

    for i, zstack in zstacks.iterrows():
        zstack = Dataset.from_flexilims(
            name=zstack.name, project=project, flexilims_session=flz_session
        )

        registered_dataset = Dataset.from_origin(
            project=project,
            origin_type="session",
            origin_id=exp_session["id"],
            dataset_type="registered_stack",
            conflicts=conflicts,
            flexilims_session=flz_session,
        )

        # sorting tifs so that they are in order of acquisition
        zstack_tifs = zstack.tif_files
        zstack_tifs.sort()

        zstack_tifs = [str(zstack.path_full / tif) for tif in zstack_tifs]
        registered_stack, nz, nchannels, _, _ = register_zstack(
            zstack_tifs, ch_to_align
        )

        # create directory for output, if it does not already exist
        if not registered_dataset.path_full.is_dir():
            os.makedirs(str(registered_dataset.path_full))

        # write registered stack to file
        with TiffWriter(
            registered_dataset.path_full.joinpath(zstack.dataset_name).with_suffix(
                ".tif"
            )
        ) as tif:
            for iplane in range(nz):
                for ich in range(nchannels):
                    tif.write(
                        np.int16(registered_stack[:, :, ich, iplane]), contiguous=True
                    )
        registered_dataset.update_flexilims(mode="overwrite")


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
    ops["save_path0"] = str(suite2p_dataset.path_full)
    # assume frame rates are the same for all recordings
    ops["fs"] = parse_si_metadata(datapaths[0])[
        "SI.hRoiManager.scanVolumeRate"
    ]  # in case of multiplane recording
    # run suite2p
    db = {"data_path": datapaths}
    opsEnd = run_s2p(ops=ops, db=db)
    if 'date_proc' in opsEnd:
        opsEnd['date_proc'] = opsEnd['date_proc'].isoformat()
    # update the database
    ops = ops.copy()
    for k, v in opsEnd.items():
        if isinstance(v, np.ndarray):
            print(f"{k} is a numpy array, skipping")
            continue
        if isinstance(v, datetime.datetime):
            ops[k] = v.strftime(r"%Y-%m-%d %H:%M:%S")
        else:
            ops[k] = v
    suite2p_dataset.extra_attributes = ops
    suite2p_dataset.update_flexilims(mode="overwrite")
    return suite2p_dataset


def dFF(f, n_components=2, verbose=True):
    """
    Helper function for calculating dF/F from raw fluorescence trace.
    Args:
        f (numpy.ndarray): shape nrois x time, raw fluorescence trace for all rois extracted from suite2p
        n_components (int): number of components for GMM. default 2.
        verbose (bool): display progress or not. Default True.

    Returns:
        dffs (numpy.ndarray): shape nrois x time, dF/F for all rois extracted from suite2p

    """
    f0 = np.zeros(f.shape[0])
    for i in range(f.shape[0]):
        gmm = mixture.GaussianMixture(n_components=n_components, random_state=42).fit(
            f[i].reshape(-1, 1)
        )
        gmm_means = np.sort(gmm.means_[:, 0])
        f0[i] = gmm_means[0]
        if verbose:
            if i % 100 == 0:
                print(f"{i}/{f.shape[0]}", flush=True)
    f0 = f0.reshape(-1, 1)
    dff = (f - f0) / f0
    return dff, f0


def calculate_dFF(
    suite2p_dataset,
    iplane,
    n_components=2,
    verbose=True,
    ast_neuropil=True,
    neucoeff=0.7,
):
    """
    Calculate dF/F for the whole session with concatenated recordings after neuropil correction.

    Args:
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        iplane (int): which plane.
        n_components (int): number of components for GMM. default 2.
        verbose (bool): display progress or not. Default True.
        ast_neuropil (bool): whether to use ASt neuropil correction or not. Default True.
        neucoeff (float): coefficient for neuropil correction. Only used if ast_neuropil
            is False. Default 0.7.

    """
    # Load fluorescence traces
    dir_path = suite2p_dataset.path_full / "suite2p" / f"plane{iplane}"
    if ast_neuropil:
        F = np.load(dir_path / "Fast.npy")
    else:
        F = np.load(dir_path / "F.npy")
        Fneu = np.load(dir_path / "Fneu.npy")
        F = F - neucoeff * Fneu
    # Calculate dFFs and save to the suite2p folder
    dff, f0 = dFF(F, n_components=n_components, verbose=verbose)
    np.save(dir_path / "dff_ast.npy" if ast_neuropil else dir_path / "dff.npy", dff)
    np.save(dir_path / "f0_ast.npy" if ast_neuropil else dir_path / "f0.npy", f0)


def spike_deconvolution_suite2p(
    suite2p_dataset, iplane, baseline="maximin", sig_baseline=10.0, win_baseline=60.0
):
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
    Fast_path = suite2p_dataset.path_full / "suite2p" / f"plane{iplane}" / "Fast.npy"
    ops_path = suite2p_dataset.path_full / "suite2p" / f"plane{iplane}" / "ops.npy"
    Fast = np.load(Fast_path)
    ops = np.load(ops_path, allow_pickle=True).tolist()

    # Params for computing and subtracting baseline
    # take the running max of the running min after smoothing with gaussian
    ops["baseline"] = baseline
    # in bins, standard deviation of gaussian with which to smooth
    ops["sig_baseline"] = sig_baseline
    # in seconds, window in which to compute max/min filters
    ops["win_baseline"] = win_baseline

    # baseline operation
    Fast = dcnv.preprocess(
        F=Fast,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
    )

    # get spikes
    spks_ast = dcnv.oasis(
        F=Fast, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"]
    )
    spks_ast_path = (
        suite2p_dataset.path_full / "suite2p" / f"plane{iplane}" / "spks_ast.npy"
    )
    np.save(spks_ast_path, spks_ast)
    np.save(ops_path, ops)


def split_recordings(flz_session, suite2p_dataset, conflicts, iplane):
    """
    suite2p concatenates all the recordings in a given session into a single file.
    To facilitate downstream analyses, we cut them back into chunks and add them
    to flexilims as children of the corresponding recordings.

    Args:
        flz_session (Flexilims): flexilims session
        suite2p_dataset (Dataset): dataset containing concatenated recordings
            to split
        conflicts (str): defines behavior if recordings have already been split
        iplane (int): which plane.

    """
    # load the ops file to find length of individual recordings
    ops_path = suite2p_dataset.path_full / "suite2p" / f"plane{iplane}" / "ops.npy"
    ops = np.load(ops_path, allow_pickle=True).tolist()
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
    # load processed data
    for iplane in range(ops["nplanes"]):
        plane_path = suite2p_dataset.path_full / "suite2p" / f"plane{iplane}"
        F, Fneu, spks = (
            np.load(plane_path / "F.npy"),
            np.load(plane_path / "Fneu.npy"),
            np.load(plane_path / "spks.npy"),
        )
        datasets_out = []
        if suite2p_dataset.extra_attributes["ast_neuropil"]:
            Fast, dff_ast, spks_ast = (
                np.load(plane_path / "Fast.npy"),
                np.load(plane_path / "dff_ast.npy"),
                np.load(plane_path / "spks_ast.npy"),
            )

        for dataset, recording_id, start, end, frame_rate in zip(
            datapaths, recording_ids, first_frames, last_frames, frame_rates
        ):
            split_dataset = Dataset.from_origin(
                project=suite2p_dataset.project,
                origin_type="recording",
                origin_id=recording_id,
                dataset_type="suite2p_traces",
                conflicts=conflicts,
            )

            if (
                split_dataset.get_flexilims_entry() is not None
            ) and conflicts == "skip":
                print(f"Dataset {split_dataset.name} already split... skipping...")
                datasets_out.append(split_dataset)
                continue
            # otherwise lets split it
            try:
                os.makedirs(str(split_dataset.path_full))
            except OSError:
                print(
                    "Error creating directory {}".format(str(split_dataset.path_full))
                )
            np.save(split_dataset.path_full / "F.npy", F[:, start:end])
            np.save(split_dataset.path_full / "Fneu.npy", Fneu[:, start:end])
            np.save(split_dataset.path_full / "spks.npy", spks[:, start:end])
            if suite2p_dataset.extra_attributes["ast_neuropil"]:
                np.save(split_dataset.path_full / "Fast.npy", Fast[:, start:end])
                np.save(split_dataset.path_full / "dff_ast.npy", dff_ast[:, start:end])
                np.save(
                    split_dataset.path_full / "spks_ast.npy",
                    spks_ast[:, start:end],
                )
            split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
            split_dataset.extra_attributes["fs"] = frame_rate
            split_dataset.update_flexilims(mode="overwrite")
            datasets_out.append(split_dataset)
        return datasets_out


def main(
    project,
    session_name,
    *,
    conflicts=None,
    run_neuropil=False,
    run_split=False,
    tau=0.7,
    nplanes=1,
    dff_ncomponents=2,
):
    """
    Process all the 2p datasets for a given session

    Args:
        project (str): name of the project, e.g. '3d_vision'
        session_name (str): name of the session
        conflicts (str): how to treat existing processed data
        run_neuropil (bool): whether or not to run neuropil extraction with the
            ASt model
        run_split (bool): whether or not to run splitting for different folders
        tau (float): time constant
        nplanes (int): number of planes
        dff_ncomponents (int): number of components for dff calculation

    """
    # get session info from flexilims
    print("Connecting to flexilims...")
    flz_session = flz.get_flexilims_session(project)
    # suite2p
    ops = default_ops()
    ops["ast_neuropil"] = run_neuropil
    ops["tau"] = tau
    ops["nplanes"] = nplanes
    ops["dff_ncomponents"] = dff_ncomponents
    
    # ops['anatomical_only'] = 2
    ops['threshold_scaling'] = 0.5
    ops['denoise'] = 1
    ops['sparse_mode'] = 1
    
    print("Running suite2p...", flush=True)
    suite2p_dataset = run_extraction(flz_session, project, session_name, conflicts, ops)

    for iplane in range(ops["nplanes"]):
        if ops["ast_neuropil"]:
            print("Running ASt neuropil correction...")
            correct_neuropil(
                suite2p_dataset.path_full / "suite2p" / ("plane" + str(iplane))
            )
        print("Calculating dF/F...")
        calculate_dFF(
            suite2p_dataset,
            iplane,
            n_components=ops["dff_ncomponents"],
            verbose=True,
            ast_neuropil=ops["ast_neuropil"],
            neucoeff=ops["neucoeff"],
        )
        if ops["ast_neuropil"]:
            print("Deconvolve spikes from neuropil corrected trace...")
            spike_deconvolution_suite2p(suite2p_dataset, iplane)
        if run_split:
            print("Splitting recordings...")
            split_recordings(
                flz_session, suite2p_dataset, conflicts="append", iplane=iplane
            )


def entry_point():
    defopt.run(main)


if __name__ == "__main__":
    defopt.run(main)
