from .ast_model import ast_model
import numpy as np
import sys
import os

def correct_neuropil(dpath):
    Fr = np.load(os.path.join(dpath, 'F.npy'))
    Fn = np.load(os.path.join(dpath, 'Fneu.npy'))
    stat = np.load(os.path.join(dpath, 'stat.npy'), allow_pickle=True)

    print('Starting neuropil correction with ASt method...', flush=True)
    traces, var_params, elbos = [], [], []
    for i, (_Fr, _Fn, _stat) in enumerate(zip(Fr, Fn, stat)):
        print('Running correction for ROI {} of {}'.
              format(i, len(Fr)), flush=True)
        trace, param, elbo = ast_model(
            np.vstack([_Fr, _Fn]),
            np.array([_stat['npix'], _stat['neuropil_mask'].shape[0]])
        )
        traces.append(trace)
        var_params.append(param)
        elbos.append(elbo)

    print('Neuropil correction completed... Saving...', flush=True)
    Fast = np.vstack(traces)
    np.save(os.path.join(dpath, 'Fast.npy'), Fast, allow_pickle=True)
    np.save(os.path.join(dpath,'ast_stat.npy'), np.vstack(var_params), allow_pickle=True)
    np.save(os.path.join(dpath,'ast_elbo.npy'), np.vstack(elbos), allow_pickle=True)    
    return Fast