import numpy as np
import sys
import defopt
import os
import flexiznam as flz
from flexiznam.schema import Dataset
from pathlib import Path
import pandas as pd
from neuropil import correct_neuropil
import itertools

from suite2p import run_s2p, default_ops


def get_paths(project, dataset_path):
    # root directory for both raw and processed data
    path_root = Path(flz.config.PARAMETERS['projects_root'])
    path_full = path_root / 'processed' / dataset_path
    return path_root, path_full

def run_extraction(flz_session, project, session_name, conflicts, ops):
    """
    Fetch data from flexilims and run suite2p with the provided settings

    Args:
        flz_session (Flexilims): flexilims session
        project (str): name of the project, determines save path
        mouse (str): name of the mouse, determines save path
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
    path_root, dataset_path = get_paths(project, suite2p_dataset.path)
    ops['save_path0'] = str(dataset_path)
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
    path_root, suite2p_dataset_path = get_paths(
        suite2p_dataset.project,
        suite2p_dataset_path.path
    )
    ops_path = suite2p_dataset_path / 'suite2p' / 'plane0' / 'ops.npy'
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
        np.load(str(savepath / 'suite2p' / 'plane0' / 'F.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'Fneu.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'spks.npy')),
        )
    datasets_out = []
    if suite2p_dataset.extra_attributes['ast_neuropil']:
        ast_path = savepath / 'suite2p' / 'plane0' / 'Fast.npy'
        Fast = np.load(str(ast_path))
    for (dataset, recording_id, start, end) in zip(datapaths, recording_ids, first_frames, last_frames):
        split_dataset = Dataset.from_origin(
            project=project,
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
        path_root, dataset_path = get_paths(project, split_dataset.path)
        try:
            os.mkdir(str(dataset_path))
        except OSError:
            print('Error creating directory {}'.format(str(dataset_path)))
        np.save(str(dataset_path / 'F.npy'), F[:,start:end])
        np.save(str(dataset_path / 'Fneu.npy'), Fneu[:,start:end])
        np.save(str(dataset_path / 'spks.npy'), spks[:,start:end])
        if suite2p_dataset.extra_attributes['ast_neuropil']:
            np.save(str(dataset_path / 'Fast.npy'), Fast[:,start:end])
        split_dataset.extra_attributes = suite2p_dataset.extra_attributes.copy()
        split_dataset.update_flexilims(mode='overwrite')
        datasets_out.append(split_dataset)
    return datasets_out


def main(project, mouse, session_name, *, conflicts=None, run_neuropil=False):
    """
    Process all the 2p datasets for a given session

    :param str project: name of the project, e.g. '3d_vision'
    :param str mouse: name of the mouse, e.g. PZAJ2.1c
    :param str session_name: name of the session
    :param str conflicts: how to treat existing processed data
    :param bool run_neuropil: whether or not to run neuropil extraction with the
        ASt model
    """
    # get session info from flexilims
    print('Connecting to flexilims...')
    flz_session = flz.get_flexilims_session(project)
    # suite2p
    ops = default_ops()
    ops['ast_neuropil'] = run_neuropil
    print('Running suite2p...')
    suite2p_dataset = run_extraction(flz_session, project, session_name, conflicts, ops)
    # neuropil correction
    if ops['ast_neuropil']:
        _, suite2p_dataset_path = get_paths(
            suite2p_dataset.project,
            suite2p_dataset.path
        )
        correct_neuropil(str(suite2p_dataset_path / 'suite2p' / 'plane0'))
    print('Splitting recordings...')
    split_recordings(flz_session, suite2p_dataset, conflicts)

def entry_point():
    defopt.run(main)

if __name__ == '__main__':
    defopt.run(main)
