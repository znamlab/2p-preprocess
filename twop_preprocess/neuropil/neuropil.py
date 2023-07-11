from .ast_model import ast_model
import numpy as np
from tqdm import tqdm


def correct_neuropil(suite2p_dataset, iplane):
    dpath = suite2p_dataset.path_full / f"plane{iplane}"

    Fr = np.load(dpath / "F.npy")
    Fn = np.load(dpath / "Fneu.npy")
    stat = np.load(dpath / "stat.npy", allow_pickle=True)

    print("Starting neuropil correction with ASt method...", flush=True)
    traces, var_params, elbos = [], [], []
    for i, (_Fr, _Fn, _stat) in enumerate(tqdm(zip(Fr, Fn, stat))):
        trace, param, elbo = ast_model(
            np.vstack([_Fr, _Fn]),
            np.array([_stat["npix"], _stat["neuropil_mask"].shape[0]]),
        )
        traces.append(trace)
        var_params.append(param)
        elbos.append(elbo)

    print("Neuropil correction completed... Saving...", flush=True)
    Fast = np.vstack(traces)
    np.save(dpath / "Fast.npy", Fast, allow_pickle=True)
    np.save(dpath / "ast_stat.npy", np.vstack(var_params), allow_pickle=True)
    np.save(dpath / "ast_elbo.npy", np.vstack(elbos), allow_pickle=True)
    return Fast
