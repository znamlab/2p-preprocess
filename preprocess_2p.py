#!/usr/bin/env python3
import numpy as np
import sys
import defopt
import flexiznam as flz
from pathlib import Path
import pandas as pd
from neuropil import correct_neuropil

from suite2p import run_s2p, default_ops

def extract_session(project, mouse, session_name):
    """
    Fetch data from flexilims and runs suite2p
    """
    flz_session = flz.get_session(project)
    exp_session = flz.get_entities(
        datatype='session', name=session_name, session=flz_session)
    if not len(exp_session):
        print('Session {} not found!'.format(session_name))
        return

    recordings = flz.get_entities(datatype='recording',
                                  origin_id=exp_session['id'][0],
                                  query_key='recording_type',
                                  query_value='two_photon',
                                  session=flz_session)
    print('Found {} two-photon recordings in session {}'.format(
        len(recordings), session_name
    ))

    path_root = Path(flz.config.PARAMETERS['projects_root'])
    datapaths = []
    for recording in recordings:
        datasets = flz.get_entities(datatype='dataset',
                         origin_id=recording['id'],
                         query_key='dataset_type',
                         query_value='scanimage',
                         session=flz_session)
        for dataset in datasets:
            this_path = path_root / dataset['path']
            if this_path.is_dir():
                datapaths.append(str(this_path))
            else:
                print('{} is not a directory'.format(this_path))

    ops = default_ops()

    savepath = Path(path_root) / project / 'processed' / mouse / session_name
    ops['save_path0'] = str(savepath)
    db = {'data_path': datapaths}
    opsEnd = run_s2p(ops=ops, db=db)
    return opsEnd, savepath


def main(project, mouse, session_name):
    """
    Process all the 2p datasets for a given session

    :param str project: name of the project, e.g. '3d_vision'
    :param str mouse: name of the mouse, e.g. PZAJ2.1c
    :param str session_date: date of the session to process in YYYY-MM-DD format
    :param int session_num: number of the session, 0-based
    """
    ops, savepath = extract_session(project, mouse, session_name)

    correct_neuropil(str(savepath / 'suite2p' / 'plane0'))

    assert len(datasets)==len(ops['frames_per_folder'])
    last_frames = np.cumsum(ops['frames_per_folder'])
    first_frames = np.concatenate([0], last_frame[:-1])

    F, Fneu, spks, Fast = (
        np.load(str(savepath / 'suite2p' / 'plane0' / 'F.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'Fneu.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'spks.npy')),
        np.load(str(savepath / 'suite2p' / 'plane0' / 'Fast.npy')),
        )
    for (dataset, start, end) in zip(datasets, first_frames, last_frames):


if __name__ == '__main__':
    defopt.run(main)
