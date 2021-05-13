#!/usr/bin/env python3
import numpy as np
import sys
import defopt
import os
import flexiznam as flz
from pathlib import Path
import pandas as pd
from neuropil import correct_neuropil

from suite2p import run_s2p, default_ops
from enum import Enum

class Conflicts(Enum):
    none = None
    overwrite = 'overwrite'
    append = 'append'
    skip = 'skip'

def run_extraction(flz_session, exp_session, conflicts, path_root, savepath, project):
    recordings = flz.get_entities(datatype='recording',
                                  origin_id=exp_session['id'][0],
                                  query_key='recording_type',
                                  query_value='two_photon',
                                  session=flz_session)
    print('Found {} two-photon recordings in session {}'.format(
        len(recordings), exp_session['name'][0]
    ))

    datapaths = []
    recording_ids = []
    for recording_id in recordings['id']:
        datasets = flz.get_entities(datatype='dataset',
                         origin_id=recording_id,
                         query_key='dataset_type',
                         query_value='scanimage',
                         session=flz_session)
        for dataset_path in datasets['path']:
            this_path = path_root / project / dataset_path
            if this_path.is_dir():
                datapaths.append(str(this_path))
                recording_ids.append(recording_id)
            else:
                print('{} is not a directory'.format(this_path))

    # run suite2p
    ops = default_ops()
    ops['save_path0'] = str(savepath)
    ops['ast_neuropil'] = False
    db = {'data_path': datapaths}

    # check if processed dataset already exists
    processed = flz.get_entities(session=flz_session,
                                 datatype='dataset',
                                 origin_id=exp_session['id'][0],
                                 query_key='dataset_type',
                                 query_value='suite2p_rois')
    already_processed = len(processed)>0
    if already_processed:
        if conflicts.value is None:
            raise flz.errors.NameNotUniqueException(
                'Session {} already processed'.format(exp_session['name'][0]))
        elif conflicts.value is 'skip':
            print('Session {} already processed... skipping extraction...'.format(exp_session['name'][0]))
            ops_file = path_root / processed['path'][0] / 'suite2p' / 'plane0' / 'ops.npy'
            opsEnd = np.load(str(ops_file), allow_pickle=True).tolist()
            return ops, opsEnd, datapaths, recording_ids
        elif conflicts.value is 'append':
            # TODO create a new version
            raise NotImplementedError('Appending not yet supported')

    opsEnd = run_s2p(ops=ops, db=db)
    if already_processed and conflicts is 'overwrite':
        # TODO update attributes
        pass
    else:
        flz.add_dataset(parent_id=exp_session['id'][0],
                        dataset_type='suite2p_rois',
                        created='',
                        path=str(savepath.relative_to(path_root)),
                        is_raw='no',
                        project_id=project,
                        session=flz_session,
                        dataset_name=exp_session['name'][0]+'_suite2p_rois',
                        attributes=ops)
    return ops, opsEnd, datapaths, recordings_ids


def split_recordings(datapaths, ops, frames_per_folder, recording_ids, flz_session, path_root, savepath):
    # split into individual recordings
    assert len(datapaths)==len(frames_per_folder)
    last_frames = np.cumsum(frames_per_folder)
    first_frames = np.concatenate(([0], last_frames[:-1]))
    F, Fneu, spks = (
        np.load(str(savepath / 'suite2p' / 'plane0' / 'F.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'Fneu.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'spks.npy')),
        )
    for (dataset, recording_id, start, end) in zip(datapaths, recording_ids, first_frames, last_frames):
        already_processed = len(flz.get_entities(session=flz_session,
                                                 datatype='dataset',
                                                 origin_id=recording_id,
                                                 query_key='dataset_type',
                                                 query_value='suite2p_traces'))>0
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
        if ops['ast_neuropil']:
            np.save(str(dataset_path / 'Fast.npy'), Fast[:,start:end])
        if not already_processed:
            flz.add_dataset(parent_id=recording_id,
                            dataset_type='suite2p_traces',
                            created='',
                            path=str(dataset_path.relative_to(path_root)),
                            is_raw='no',
                            session=flz_session,
                            dataset_name=dataset_dir+'_suite2p_traces',
                            attributes=ops)


def main(project, mouse, session_name, *, conflicts=Conflicts.none):
    """
    Process all the 2p datasets for a given session

    :param str project: name of the project, e.g. '3d_vision'
    :param str mouse: name of the mouse, e.g. PZAJ2.1c
    :param str session_name: name of the session
    :param Conflicts conflicts: how to treat existing processed data
    """
    # root directory for both raw and processed data
    path_root = Path(flz.config.PARAMETERS['projects_root'])
    savepath = Path(path_root) / project / 'processed' / mouse / session_name
    # get session info from flexilims
    flz_session = flz.get_session(project)
    exp_session = flz.get_entities(
        datatype='session', name=session_name, session=flz_session)
    if not len(exp_session):
        print('Session {} not found!'.format(session_name))
        return

    ops, opsEnd, datapaths, recording_ids = run_extraction(
        flz_session, exp_session, conflicts, path_root, savepath, project
        )
    # neuropil correction
    if ops['ast_neuropil']:
        Fast = correct_neuropil(str(savepath / 'suite2p' / 'plane0'))

    split_recordings(datapaths, ops, opsEnd['frames_per_folder'],
                     recording_ids, flz_session, path_root, savepath)


if __name__ == '__main__':
    defopt.run(main)
