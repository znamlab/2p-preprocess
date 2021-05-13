from .ast_model import ast_model
import numpy as np
import multiprocessing as mp
import sys
import os

def correct_neuropil(dpath):
    Fr = np.load(os.path.join(dpath, 'F.npy'))
    Fn = np.load(os.path.join(dpath, 'Fneu.npy'))
    stat = np.load(os.path.join(dpath, 'stat.npy'), allow_pickle=True)

    cores = mp.cpu_count()
    print('Starting parallel pool with {} processes...'.format(cores))
    pool = mp.Pool(cores)

    print('Starting neuropil correction with ASt method...')
    results = [pool.apply(ast_model, args=(
                    np.vstack([_Fr, _Fn]),
                    np.array([_stat['npix'], _stat['neuropil_mask'].shape[0]])
                    )) for (_Fr, _Fn, _stat) in zip(Fr, Fn, stat)]

    pool.close()

    traces, var_params = [], []

    for (trace, param) in results:
        traces.append(trace)
        var_params.append(param)

    print('Neuropil correction completed... Saving...')
    Fast = np.vstack(traces)
    np.save(os.path.join(dpath, 'Fast.npy'), Fast, allow_pickle=True)
    np.save(dpath + 'ast_stat.npy', np.vstack(var_params), allow_pickle=True)
    return Fast

def main():
    datapath = sys.argv[1]
    correct_neuropil(datapath)
