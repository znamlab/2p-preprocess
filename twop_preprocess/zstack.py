import numpy as np
import os
import flexiznam as flz
from flexiznam.schema import Dataset
import itertools
from tifffile import TiffFile, TiffWriter
from itertools import chain
from more_itertools import chunked, distribute
import scipy.fft as fft
from tqdm import tqdm
from twop_preprocess.utils import parse_si_metadata, load_ops
from functools import partial


print = partial(print, flush=True)


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


def register_zstack(tiff_paths, ops):
    """
    Apply motion correction to a z-stack.

    We first apply motion correction to individual frames in each slice and the
    register adjacent slices to each other.

    Args:
        tiff_paths (list): list of full paths to the z-stack files
        ops (dict): dictionary of ops

    Rerurns:
        numpy.ndarray: z-stack after applying motion correction (X x Y x Z)
        nz: int, number of slices in aligned_stack
        nchannels: int, number of channels in aligned_stack

    """
    # get the aquisition params from the first .tif file
    si_dict = parse_si_metadata(tiff_paths[0])
    if isinstance(si_dict["SI.hChannels.channelSave"], int):
        nchannels = si_dict["SI.hChannels.channelSave"]
        ops["ch_to_align"] = 0
    else:
        nchannels = len(si_dict["SI.hChannels.channelSave"])
        assert nchannels > ops["ch_to_align"]
    # get a list of stacks from the acquisition
    stack_list = [TiffFile(tiff_path) for tiff_path in tiff_paths]
    # chain the the pages from the stack
    stack_pages = itertools.chain(*[stack.pages for stack in stack_list])
    nframes = int(si_dict["SI.hStackManager.framesPerSlice"])

    chunk_size = nframes * nchannels

    nx = int(si_dict["SI.hRoiManager.pixelsPerLine"])
    ny = int(si_dict["SI.hRoiManager.linesPerFrame"])
    nz = int(si_dict["SI.hStackManager.actualNumSlices"])

    # if the zstack is split into multiple acquisitions to be concatenated, then get the total number of z planes
    if ops["zstack_concat"]:
        for dataset in ops["dataset_name"]:
            # get the ScanImage acquisition string for each dataset
            si_acquisition = "_" + dataset.split("_")[-1] + "_"
            if si_acquisition in str(tiff_paths[0]):
                pass
            else:
                tiff_paths_subset = [
                    tiff_path
                    for tiff_path in tiff_paths
                    if si_acquisition in str(tiff_path)
                ]
                tmp = parse_si_metadata(tiff_paths_subset[0])
                nz += int(tmp["SI.hStackManager.actualNumSlices"])

    registered_stack = np.zeros((nx, ny, nchannels, nz))

    # process stack one slice at a time (for sequentially imaged volumes)
    if ops["sequential_volumes"]:
        print("Registering stack as a sequence of volumes", flush=True)
        nvolumes = int(si_dict["SI.hStackManager.actualNumVolumes"])
        iterate_over = distribute(nz, chunked(stack_pages, chunk_size))
    else:
        nvolumes = nframes
        iterate_over = chunked(stack_pages, chunk_size)

    frame_shifts = np.zeros((nvolumes, 2, nz))
    for iplane, plane in tqdm(
        enumerate(iterate_over),
        total=nz,
        desc="Imaging planes",
    ):
        if ops["sequential_volumes"]:
            data = np.asarray([page.asarray() for frame in plane for page in frame])
        else:
            data = np.asarray([page.asarray() for page in plane])
        if ops["bidi_correction"]:
            if iplane == 0:
                bidi_shift = estimate_bidi_correction(data[ops["ch_to_align"], :, :])
                print(f"Estimated bidi shift: {bidi_shift}", flush=True)
            data = apply_bidi_correction(data, bidi_shift)
        # generate reference image for the current slice
        for i in range(ops["iter"]):
            if i == 0:
                if ops["pick_ref"]:
                    ref_frames = data[ops["ch_to_align"] :: nchannels, :, :]
                    m = np.reshape(ref_frames, (nvolumes, -1))
                    c = np.sum(np.corrcoef(m), axis=1)
                    good_frames = c > np.percentile(c, ops["pick_ref_percentile"])
                    template_image = np.mean(ref_frames[good_frames, :, :], axis=0)
                else:
                    template_image = np.mean(
                        data[ops["ch_to_align"] :: nchannels, :, :], axis=0
                    )
            else:
                template_image = registered_stack[
                    :, :, ops["ch_to_align"], iplane
                ].copy()
                registered_stack[:, :, :, iplane] = 0
            template_image_fft = fft.fft2(template_image)
            # use reference image to align individual planes
            for iframe in tqdm(range(nvolumes), leave=False, desc="Frames"):
                shifts, cc = phase_corr(
                    template_image_fft,
                    data[nchannels * iframe + ops["ch_to_align"], :, :],
                    max_shift=ops["max_shift"],
                    whiten=False,
                    fft_ref=False,
                )
                frame_shifts[iframe, :, iplane] = shifts
                for ich in range(nchannels):
                    registered_stack[:, :, ich, iplane] += np.roll(
                        data[nchannels * iframe + ich, :, :],
                        (int(shifts[0]), int(shifts[1])),
                        axis=(0, 1),
                    )

    plane_shifts = np.zeros((2, nz))
    if not ops["align_planes"]:
        return (
            registered_stack / int(nvolumes),
            nz,
            nchannels,
            frame_shifts,
            plane_shifts,
        )
    aligned_stack = np.zeros((nx, ny, nchannels, nz))
    # we don't need to align the very first plane
    aligned_stack[:, :, :, 0] = registered_stack[:, :, :, 0]

    # align planes to each other
    print("Aligning planes to each other", flush=True)
    for iplane in range(1, nz):
        previous_shifts = plane_shifts[:, iplane - 1]
        target = np.roll(
            registered_stack[:, :, ops["ch_to_align"], iplane],
            (int(previous_shifts[0]), int(previous_shifts[1])),
            axis=(0, 1),
        )
        shifts = phase_corr(
            aligned_stack[:, :, ops["ch_to_align"], iplane - 1],
            target,
            max_shift=ops["max_shift"],
            whiten=True,
            fft_ref=True,
        )[0]
        plane_shifts[:, iplane] = shifts
        for ich in range(nchannels):
            aligned_stack[:, :, ich, iplane] = np.roll(
                registered_stack[:, :, ich, iplane],
                (
                    int(shifts[0] + previous_shifts[0]),
                    int(shifts[1] + previous_shifts[1]),
                ),
                axis=(0, 1),
            )
            if shifts[0] > 0:
                aligned_stack[: int(shifts[0]), :, ich, iplane] = 0
            elif shifts[0] < 0:
                aligned_stack[int(shifts[0]) :, :, ich, iplane] = 0
            if shifts[1] > 0:
                aligned_stack[:, : int(shifts[1]), ich, iplane] = 0
            elif shifts[1] < 0:
                aligned_stack[:, int(shifts[1]) :, ich, iplane] = 0
    if ops["align_planes"] and ops["sequential_volumes"]:
        return (
            aligned_stack / int(nvolumes),
            nz,
            nchannels,
            frame_shifts,
            plane_shifts,
        )
    return aligned_stack / int(nframes), nz, nchannels, frame_shifts, plane_shifts


def run_zstack_registration(
    project, session_name, datasets=None, conflicts="append", ops={}
):
    """
    Apply motion correction to all zstacks for a single session, create Flexylims
    entries for each registered zstacks

    Args:
        project (str): human-readable string for project name in Flexylims
                (hexadecimal id fails with Dataset.from_origin)
        session_name (str): string matching Flexylims session name
        conflicts (str): string for handling flexilims conflicts, if more than
                one zstack needs to be registered for session, use conflicts="append"
        ops (dict): dictionary of ops

    """
    ops = load_ops(ops, zstack=True)
    print(f"Regisering zstacks for session {session_name} from project {project}")
    print(f"Using ops: {ops}")
    print("Connecting to flexilims...")
    flz_session = flz.get_flexilims_session(project)
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
    zstacks.reset_index(drop=True, inplace=True)
    if datasets is not None:
        assert np.all([dataset in zstacks["name"].values for dataset in datasets])
        zstacks = zstacks[zstacks["name"].isin(datasets)]

    all_zstack_tifs = []

    for i, row in zstacks.iterrows():
        print(f"Registering {row['name']}")
        zstack = Dataset.from_flexilims(
            name=row['name'], project=project, flexilims_session=flz_session
        )
        # sorting tifs so that they are in order of acquisition
        zstack.tif_files.sort()
        zstack_tifs = [zstack.path_full / tif for tif in zstack.tif_files]

        if ops["zstack_concat"]:
            all_zstack_tifs.extend(zstack_tifs)
            if i < zstacks.shape[0] - 1:
                continue
            registered_stack, nz, nchannels, frame_shifts, plane_shifts = (
                register_zstack(all_zstack_tifs, ops)
            )
        else:
            registered_stack, nz, nchannels, frame_shifts, plane_shifts = (
                register_zstack(zstack_tifs, ops)
            )

        registered_dataset = Dataset.from_origin(
            project=project,
            origin_type="session",
            origin_id=exp_session["id"],
            dataset_type="registered_stack",
            base_name=zstack.dataset_name + "_registered",
            conflicts=conflicts,
            flexilims_session=flz_session,
        )

        registered_dataset.path = registered_dataset.path.with_suffix(".tif")

        if not registered_dataset.path_full.parent.exists():
            os.makedirs(registered_dataset.path_full.parent)

        registered_stack = np.clip(np.copy(registered_stack), a_min=0, a_max=None)
        # write registered stack to file
        with TiffWriter(registered_dataset.path_full) as tif:
            for iplane in range(nz):
                for ich in range(nchannels):
                    tif.write(
                        np.int16(registered_stack[:, :, ich, iplane]), contiguous=True
                    )
        np.savez(
            registered_dataset.path_full.with_suffix(".npz"),
            frame_shifts=frame_shifts,
            plane_shifts=plane_shifts,
            allow_pickle=True,
        )
        registered_dataset.extra_attributes = ops
        registered_dataset.update_flexilims(mode="overwrite")
