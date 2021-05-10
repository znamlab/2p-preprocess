#!/usr/bin/env python3
import numpy as np
import sys
import defopt
import flexiznam as flz
from pathlib import Path
import pandas as pd

from suite2p import run_s2p, default_ops

def main(project, mouse, session_date, session_num=0):
    """
    Process all the 2p datasets for a given session

    :param str project: name of the project, e.g. '3d_vision'
    :param str mouse: name of the mouse, e.g. PZAJ2.1c
    :param str session_date: date of the session to process in YYYY-MM-DD format
    :param int session_num: number of the session, 0-based
    """
    flz_session = flz.get_session(project)
    session_name = mouse + '_' + session_date + '_' + str(session_num)
    exp_session = flz.get_entities(
        datatype='session', name='session_name', session=flz_session)
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
                datapaths.append(this_path)
            else:
                print('{} is not a directory'.format(this_path))

    ops = default_ops()
    db = {'data_path': datapaths}

    opsEnd = run_s2p(ops=ops, db=db[0])

if __name__ == '__main__':
    defopt.run(main)
