import numpy as np
import os
import flexiznam as flz
from flexiznam.schema import Dataset
import itertools
from tifffile import TiffFile, TiffWriter
from more_itertools import chunked
import scipy.fft as fft
from tqdm import tqdm
from twop_preprocess.utils import parse_si_metadata


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
    project, session_name, conflicts="append", ch_to_align=0
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
        ch_to_align (int): channel to use for calculating shifts

    """
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

    for _, zstack in zstacks.iterrows():
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
