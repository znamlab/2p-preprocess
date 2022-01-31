import numpy as np
import sys
import defopt
import os
import re
import flexiznam as flz
from flexiznam.schema import Dataset
from pathlib import Path
import pandas as pd
from neuropil import correct_neuropil
import itertools

from suite2p import run_s2p, default_ops
from ScanImageTiffReader import ScanImageTiffReader

from tifffile import TiffFile, TiffWriter
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
from more_itertools import ichunked, chunked
import scipy.fft as fft
from suite2p.registration.rigid import phasecorr, phasecorr_reference

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
    if not tiff_path.endswith('.tif'):
        tiffs = [tiff for tiff in os.listdir(tiff_path) if tiff.endswith(".tif")]
    else:
        tiffs = [ tiff_path, ]
    if tiffs:
        tiff_path = str(Path(tiff_path) / tiffs[0])
        tiff = ScanImageTiffReader(tiff_path)
        # list of SI metadata keywords to export
        si_keys = [
            'SI.hRoiManager.scanZoomFactor',
            'SI.hRoiManager.scanFramePeriod',
            'SI.hRoiManager.scanFrameRate',
            'SI.hRoiManager.scanVolumeRate',
            'SI.hStackManager.actualNumSlices',
            'SI.hStackManager.actualNumVolumes',
            'SI.hStackManager.actualStackZStepSize',
            'SI.hStackManager.framesPerSlice',
            'SI.hRoiManager.pixelsPerLine',
            'SI.hRoiManager.linesPerFrame'
        ]

        si_dict = {}
        for key in si_keys:
            val = float(re.search(f'{key} = (\d+(?:\.\d+)?)', tiff.metadata()).group(1))
            si_dict[key] = val

        channels = re.search('SI.hChannels.channelSave = \[(.*)\]', tiff.metadata()).group(1).split()
        si_dict['SI.hChannels.channelSave'] = [int(s) for s in channels]
        return si_dict
    else:
        return None


def get_frame_rate(tiff_path):
    tiffs = [tiff for tiff in os.listdir(tiff_path) if tiff.endswith(".tif")]
    if tiffs:
        tiff_path = str(Path(tiff_path) / tiffs[0])
        return float(
            re.search(
                'scanVolumeRate = (\d+\.\d+)',
                ScanImageTiffReader(tiff_path).metadata()
            ).group(1)
        )
    else:
        return None


def register_zstack(tiff_path, ch_to_align=0):
    """
    Apply motion correction to a z-stack.

    We first apply motion correction to individual frames in each slice and the
    register adjacent slices to each other.

    Args:
        tiff_path (str): path to the z-stack file
        ch_to_align (int): channel to use for registration
        nchannels (int): number of channels in the stack

    Rerurns:
        numpy.ndarray: z-stack after applying motion correction (X x Y x Z)
        nz: int, number of slices in aligned_stack
        nchannels: int, number of channels in aligned_stack
    """
    si_dict = parse_si_metadata(tiff_path)
    nchannels = len(si_dict['SI.hChannels.channelSave'])
    assert nchannels>ch_to_align
    stack = TiffFile(tiff_path)
    nframes = int(si_dict['SI.hStackManager.framesPerSlice'])

    chunk_size = nframes * nchannels

    nx = int(si_dict['SI.hRoiManager.pixelsPerLine'])
    ny = int(si_dict['SI.hRoiManager.linesPerFrame'])
    nz = int(si_dict['SI.hStackManager.actualNumSlices'])

    registered_stack = np.zeros((nx, ny, nchannels, nz))

    # process stack one slice at a time
    for (iplane, plane) in enumerate(chunked(stack.pages, chunk_size)):
        print(f'Registering plane {iplane+1} of {nz}', flush=True)
        data = np.asarray([page.asarray() for page in plane])
        # generate reference image for the current slice
        template_image = np.mean(data[ch_to_align:nchannels:,:,:], axis=0)
        template_image_fft = fft.fftn(template_image)
        # yshift, xshift, _ = phasecorr(
        #     data[ch_to_align:nchannels:,:,:],
        #     template_image_fft,
        #     0.2,
        #     0
        # )
        # use reference image to align individual planes
        for iframe in range(nframes):
            shifts = phase_cross_correlation(
                template_image_fft,
                fft.fftn(data[iframe+ch_to_align*nframes,:,:]),
                space='fourier'
            )
            for ich in range(nchannels):
                registered_stack[:,:,ich,iplane] += shift(
                    data[iframe+ich*nframes,:,:],
                    (shifts[0][0], shifts[0][1]),
                    output=None,
                    order=3,
                    mode='constant',
                    cval=0.0,
                    prefilter=True
                )

    aligned_stack = np.zeros((nx, ny, nchannels, nz))
    # we don't need to align the very first plane
    aligned_stack[:,:,:,0] = registered_stack[:,:,:,0]

    # align planes to each other
    print('Aliging planes to each other', flush=True)
    for iplane in range(1, nz):
        # it helps to do subpixel registration to align slices
        shifts = phase_cross_correlation(
            aligned_stack[:,:,ch_to_align,iplane-1],
            registered_stack[:,:,ch_to_align,iplane],
            space='real',
            upsample_factor=10
        )
        for ich in range(nchannels):
            aligned_stack[:,:,ich,iplane] = shift(
                    registered_stack[:,:,ich,iplane],
                    (shifts[0][0],shifts[0][1]),
                    output=None,
                    order=3,
                    mode='constant',
                    cval=0.0,
                    prefilter=True
            )
    return aligned_stack / int(nframes), nz, nchannels


def run_zstack_registration(flz_session, project, session_name, conflicts='append',
        ch_to_align=0):
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

    Returns:
        nothing, writes numpy.ndarray from register_zstack to a tif file

    """
    # get experimental session
    exp_session = flz.get_entity(
        datatype='session',
        name=session_name,
        flexilims_session=flz_session
    )

    # get all zstacks from session
    zstacks = flz.get_entities(
        datatype='dataset',
        origin_id=exp_session['id'],
        query_key='stack_type',
        query_value='zstack',
        flexilims_session=flz_session
    )

    for i,j in zstacks.iterrows():
        # get zstack Dataset with flexilims
        zstack = Dataset.from_flexilims(
            name=j.name,
            project=project,
            flm_session=flz_session)

        # add flm_session as argument
        registered_dataset = Dataset.from_origin(
            project=project,
            origin_type='session',
            origin_id=exp_session['id'],
            dataset_type='registered_stack',
            conflicts=conflicts,
            flm_session=flz_session
        )

        # save registered stack under the same path as raw stack + "_registered"
        # in the processed directory
        registered_dataset.path = zstack.path.joinpath("registered")
        # registered_dataset.path.stem += '_registered'

        # change registered Dataset to unique name
        registered_dataset.name = zstack.name + '_registered'

        # ensure there is only one file in zstack Dataset object
        assert len(zstack.extra_attributes['file_list']) == 1

        registered_stack, nz, nchannels = register_zstack(
            str(zstack.path_full/zstack.extra_attributes['file_list'][0]),
            ch_to_align
        )

        # create directory for output, if it does not already exist
        if not registered_dataset.path_full.is_dir():
            registered_dataset.path_full.mkdir(parents=True, exist_ok=False)

        # write registered stack to file
        with TiffWriter(registered_dataset.path_full.joinpath(zstack.extra_attributes['file_list'][0])) as tif:
            for iplane in range(nz):
                for ich in range(nchannels):
                    tif.write(
                        np.int16(registered_stack[:,:,ich,iplane]),
                        contiguous=True
                    )
        registered_dataset.update_flexilims(mode='overwrite',
                                            flm_session=flz_session)


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
        datatype='session',
        name=session_name,
        flexilims_session=flz_session
    )
    suite2p_dataset = Dataset.from_origin(
        project=project,
        origin_type='session',
        origin_id=exp_session['id'],
        dataset_type='suite2p_rois',
        conflicts=conflicts
    )
    # if already on flexilims and not re-processing, then do nothing
    if (suite2p_dataset.get_flexilims_entry() is not None) and conflicts == 'skip':
        print(
            'Session {} already processed... skipping extraction...'
            .format(exp_session['name'])
        )
        return suite2p_dataset
    # fetch SI datasets
    si_datasets = flz.get_datasets(
        exp_session['id'],
        recording_type='two_photon',
        dataset_type='scanimage',
        flexilims_session=flz_session
    )
    datapaths = []
    for _, p in si_datasets.items(): datapaths.extend(p)
    # set save path
    ops['save_path0'] = str(suite2p_dataset.path_full)
    # assume frame rates are the same for all recordings
    ops['fs'] = get_frame_rate(datapaths[0])
    # run suite2p
    db = {'data_path': datapaths}
    opsEnd = run_s2p(ops=ops, db=db)
    # update the database
    suite2p_dataset.extra_attributes = ops.copy()
    suite2p_dataset.update_flexilims(mode='overwrite')
    return suite2p_dataset


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
    ops_path = suite2p_dataset.path_full / 'suite2p' / 'plane0' / 'ops.npy'
    ops = np.load(ops_path, allow_pickle=True).tolist()
    # get scanimage datasets
    datasets = flz.get_datasets(
        suite2p_dataset.origin_id,
        recording_type='two_photon',
        dataset_type='scanimage',
        flexilims_session=flz_session
    )
    datapaths = []
    recording_ids = []
    for r, p in datasets.items():
        datapaths.extend(p)
        recording_ids.extend(itertools.repeat(r, len(p)))
    # split into individual recordings
    assert len(datapaths)==len(ops['frames_per_folder'])
    last_frames = np.cumsum(ops['frames_per_folder'])
    first_frames = np.concatenate(([0], last_frames[:-1]))
    # load processed data
    F, Fneu, spks = (
        np.load(str(suite2p_dataset.path_full / 'suite2p' / 'plane0' / 'F.npy')),
        np.load(str(suite2p_dataset.path_full / 'suite2p' / 'plane0' / 'Fneu.npy')),
        np.load(str(suite2p_dataset.path_full / 'suite2p' / 'plane0' / 'spks.npy')),
        )
    datasets_out = []
    if suite2p_dataset.extra_attributes['ast_neuropil']:
        ast_path = suite2p_dataset.path_full / 'suite2p' / 'plane0' / 'Fast.npy'
        Fast = np.load(str(ast_path))
    for (dataset, recording_id, start, end) in zip(datapaths, recording_ids, first_frames, last_frames):
        split_dataset = Dataset.from_origin(
            project=suite2p_dataset.project,
            origin_type='recording',
            origin_id=recording_id,
            dataset_type='suite2p_traces',
            conflicts=conflicts
        )
        if (split_dataset.get_flexilims_entry() is not None) and conflicts == 'skip':
            print(
                'Dataset {} already split... skipping...'
                .format(split_dataset.name)
            )
            datasets_out.append(split_dataset)
            continue
        # otherwise lets split it
        try:
            split_dataset.path_full.mkdir(parents=True)
        except OSError:
            print('Error creating directory {}'.format(str(split_dataset.path_full)))
        np.save(str(split_dataset.path_full / 'F.npy'), F[:,start:end])
        np.save(str(split_dataset.path_full / 'Fneu.npy'), Fneu[:,start:end])
        np.save(str(split_dataset.path_full / 'spks.npy'), spks[:,start:end])
        if suite2p_dataset.extra_attributes['ast_neuropil']:
            np.save(str(split_dataset.path_full / 'Fast.npy'), Fast[:,start:end])
        split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
        split_dataset.update_flexilims(mode='overwrite')
        datasets_out.append(split_dataset)
    return datasets_out


def main(project, session_name, *, conflicts=None, run_neuropil=False,
         tau=0.7):
    """
    Process all the 2p datasets for a given session

    Args:
        project (str): name of the project, e.g. '3d_vision'
        session_name (str): name of the session
        conflicts (str): how to treat existing processed data
        run_neuropil (bool): whether or not to run neuropil extraction with the
            ASt model
        tau (float): time constant
    """
    # get session info from flexilims
    print('Connecting to flexilims...')
    flz_session = flz.get_flexilims_session(project)
    # suite2p
    ops = default_ops()
    ops['ast_neuropil'] = run_neuropil
    ops['tau'] = tau
    print('Running suite2p...', flush=True)
    suite2p_dataset = run_extraction(flz_session, project, session_name, conflicts, ops)
    # neuropil correction
    if ops['ast_neuropil']:
        correct_neuropil(str(suite2p_dataset.path_full / 'suite2p' / 'plane0'))
    print('Splitting recordings...')
    split_recordings(flz_session, suite2p_dataset, conflicts)

def entry_point():
    defopt.run(main)

if __name__ == '__main__':
    defopt.run(main)
