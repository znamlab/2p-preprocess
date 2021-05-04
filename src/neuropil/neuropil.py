from .ast_model import ast_model
import numpy as np
import multiprocessing as mp
import sys

def correct_neuropil(dpath):
    Fr = np.load(dpath + 'F.npy')
    Fn = np.load(dpath + 'Fneu.npy')
    stat = np.load(dpath + 'stat.npy')

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
    np.save(dpath + 'Fast.npy', np.vstack(traces), allow_pickle=True)
    np.save(dpath + 'ast_stat.npy', np.vstack(var_params), allow_pickle=True)

def main():
    datapath = sys.argv[1]
    correct_neuropil(datapath)
