from .ast_model import ast_model
import numpy as np
from tqdm import tqdm


def correct_neuropil(suite2p_dataset, iplane):
    """
    Apply Asymmetric Student's t-model (ASt) neuropil correction to a Suite2p plane.

    This function loads the raw fluorescence (F) and neuropil (Fneu) traces for
    a specific plane, and uses the ASt model to estimate the corrected neural
    activity. The results (Fast, ast_stat, ast_elbo) are saved to the plane's directory.

    Args:
        suite2p_dataset (Dataset): Flexilims Dataset object for the Suite2p ROIs.
        iplane (int): The index of the plane to process.

    Returns:
        np.ndarray: The ASt-corrected fluorescence traces (n_rois x n_frames).
    """
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
