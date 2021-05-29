import numpy as np
import sys
import defopt
import os
import flexiznam as flz
from pathlib import Path
import pandas as pd
from neuropil import correct_neuropil
import itertools

from suite2p import run_s2p, default_ops


def get_paths(project, mouse, session_name):
    # root directory for both raw and processed data
    path_root = Path(flz.config.PARAMETERS['projects_root'])
    savepath = path_root / project / 'processed' / mouse / session_name
    return path_root, savepath

def run_extraction(flz_session, project, mouse, session_name, conflicts, ops):
    """
    Fetch data from flexilims and run suite2p with the provided settings

    :param Flexilims flz_session: flexilims session
    :param str project: name of the project, determines save path
    :param str mouse: name of the mouse, determines save path
    :param str session_name: name of the session, used to find data on flexilims
    :param str conflicts: defines behavior if recordings have already been split
    :param dict ops: dictionary of suite2p settings
    """
    path_root, savepath = get_paths(project, mouse, session_name)
    # get experimental session
    exp_session = flz.get_entity(
        datatype='session',
        name=session_name,
        flexilims_session=flz_session
    )
    # check if processed dataset already exists
    processed = flz.get_entities(
        flexilims_session=flz_session,
        datatype='dataset',
        origin_id=exp_session['id'],
        query_key='dataset_type',
        query_value='suite2p_rois'
    )
    already_processed = len(processed)>0
    if already_processed:
        if conflicts is None:
            raise flz.errors.NameNotUniqueException(
                'Session {} already processed'.format(exp_session['name']))
        elif conflicts == 'skip':
            print('Session {} already processed... skipping extraction...'.format(exp_session['name']))
            return
        elif conflicts == 'append':
            # TODO create a new version
            raise NotImplementedError('Appending not yet supported')
    # get all datasets
    datasets = flz.get_datasets(
        exp_session['id'],
        recording_type='two_photon',
        dataset_type='scanimage',
        flexilims_session=flz_session
    )
    datapaths = []
    for _, p in datasets.items(): datapaths.extend(p)

    # run suite2p
    db = {'data_path': datapaths}
    opsEnd = run_s2p(ops=ops, db=db)

    if already_processed and conflicts is 'overwrite':
        # TODO update attributes
        pass
    else:
        flz.add_dataset(
            parent_id=exp_session['id'],
            dataset_type='suite2p_rois',
            created='',
            path=str(savepath.relative_to(path_root)),
            is_raw='no',
            project_id=project,
            flexilims_session=flz_session,
            dataset_name=exp_session['name']+'_suite2p_rois',
            attributes=ops
        )


def split_recordings(flz_session, project, mouse, session_name, conflicts):
    """
    suite2p concatenates all the recordings in a given session into a single file.
    To facilitate downstream analyses, we cut them back into chunks and add them
    to flexilims as children of the corresponding recordings.

    :param Flexilims flz_session: flexilims session
    :param str project: name of the project, determines save path
    :param str mouse: name of the mouse, determines save path
    :param str session_name: name of the session, determines save path
    :param str conflicts: defines behavior if recordings have already been split
    """
    path_root, savepath = get_paths(project, mouse, session_name)
    # get experimental session
    exp_session = flz.get_entity(
        datatype='session',
        name=session_name,
        flexilims_session=flz_session
    )
    # get scanimage datasets
    datasets = flz.get_datasets(
        exp_session['id'],
        recording_type='two_photon',
        dataset_type='scanimage',
        flexilims_session=flz_session
    )
    # get preprocessed suite2p dataset
    suite2p_dataset = flz.get_entity(
        session=flz_session,
        datatype='dataset',
        origin_id=exp_session['id'],
        query_key='dataset_type',
        query_value='suite2p_rois'
    )
    # load the ops file to find length of individual recordings
    ops_path = path_root / suite2p_dataset['path'] / 'suite2p' / 'plane0' / 'ops.npy'
    ops = np.load(ops_path, allow_pickle=True).tolist()
    datasets = flz.get_datasets(
        exp_session['id'],
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
    ast_path = savepath / 'suite2p' / 'plane0' / 'Fast.npy'
    if ast_path.exists():
        Fast = np.load(str(ast_path))
    for (dataset, recording_id, start, end) in zip(datapaths, recording_ids, first_frames, last_frames):
        already_processed = len(flz.get_entities(
            flexilims_session=flz_session,
            datatype='dataset',
            origin_id=recording_id,
            query_key='dataset_type',
            query_value='suite2p_traces'
        ))>0
        # TODO what is the dataset already exists
        dataset_dir = Path(dataset).name
        dataset_path = savepath / dataset_dir
        try:
            os.mkdir(str(dataset_path))
        except OSError:
            print('Error creating directory {}'.format(str(savepath / dataset_dir)))
        np.save(str(dataset_path / 'F.npy'), F[:,start:end])
        np.save(str(dataset_path / 'Fneu.npy'), Fneu[:,start:end])
        np.save(str(dataset_path / 'spks.npy'), spks[:,start:end])
        if ast_path.exists():
            np.save(str(dataset_path / 'Fast.npy'), Fast[:,start:end])
        if not already_processed:
            flz.add_dataset(
                parent_id=recording_id,
                dataset_type='suite2p_traces',
                created='',
                path=str(dataset_path.relative_to(path_root)),
                is_raw='no',
                flexilims_session=flz_session,
                dataset_name=dataset_dir+'_suite2p_traces',
                attributes=ops
            )


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
    flz_session = flz.get_session(project)
    # suite2p
    _, savepath = get_paths(project, mouse, session_name)
    ops = default_ops()
    ops['save_path0'] = str(savepath)
    ops['ast_neuropil'] = run_neuropil
    print('Running suite2p...')
    run_extraction(flz_session, project, mouse, session_name, conflicts, ops)
    # neuropil correction
    if ops['ast_neuropil']:
        suite2p_path = str(savepath / 'suite2p' / 'plane0')
        correct_neuropil(suite2p_path)
    print('Splitting recordings...')
    split_recordings(flz_session, project, mouse, session_name, conflicts)

def entry_point():
    defopt.run(main)

if __name__ == '__main__':
    defopt.run(main)
