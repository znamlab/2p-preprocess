from tifffile import TiffFile
from pathlib import Path
import os
import yaml
import numpy as np


def parse_si_metadata(tiff_path):
    """
    Read metadata from a ScanImage TIFF file.

    Extracts the 'FrameData' metadata dictionary from the TIFF header.

    Args:
        tiff_path (str or Path): Path to a ScanImage TIFF file or a directory
            containing TIFF files.

    Returns:
        dict: Dictionary of ScanImage parameters, or None if no TIFFs found.
    """
    assert os.path.exists(tiff_path), f"Error: {tiff_path} does not exist"

    if tiff_path.suffix != ".tif":
        tiffs = [tiff_path / tiff for tiff in sorted(tiff_path.glob("*.tif"))]
    else:
        tiffs = [
            tiff_path,
        ]
    if tiffs:
        return TiffFile(tiffs[0]).scanimage_metadata["FrameData"]
    else:
        return None


def load_ops(user_ops, zstack=False):
    """
    Generate the final options (ops) dictionary for the pipeline.

    Combines Suite2p defaults, pipeline-specific defaults (from default_ops.yml),
    user configuration (~/.2p_preprocess/config.yml), and runtime overrides.

    Args:
        user_ops (dict): Dictionary of user-specified overrides provided at runtime.
        zstack (bool, optional): If True, loads settings for z-stack registration.
            Default False.

    Returns:
        dict: The final consolidated options dictionary.
    """
    from suite2p import default_ops

    default_ops_fname = Path(__file__).parent / "default_ops.yml"
    with open(default_ops_fname, "r") as f:
        pipeline_ops = yaml.safe_load(f)
    if zstack:
        ops = pipeline_ops["zstack"]
    else:
        pipeline_ops.pop("zstack", None)
        # update suite2p defaults with pipeline defaults
        suite2p_ops = default_ops()
        ops = dict(suite2p_ops, **pipeline_ops)
    # update with user specified config
    config_path = Path.home() / ".2p_preprocess" / "config.yml"
    if config_path.is_file():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        if zstack:
            config = config.get("zstack", {})
        # update with user specified config
        ops = dict(ops, **config)
    # update with user specified ops provided at runtime
    ops = dict(ops, **user_ops)
    return ops


def load_meanImg(suite2p_dataset):
    """Takes a single suite2p dataset and loads the im from the suite2p run.

    Args:
        suite2p_dataset: flexiznam.schema.datasets.Dataset object with
            suite2p output paths and run parameters

    Returns:
        numpy.ndarray of shape (n_planes, Y, X) where n_planes is the
            number of imaging planes of the recording

    """
    s2p_output_path = suite2p_dataset.path_full
    # Get the size of the image for each frame of the recording, assumes square images
    if suite2p_dataset.extra_attributes["lx"] is None or np.isnan(
        suite2p_dataset.extra_attributes["lx"]
    ):
        n_px = int(suite2p_dataset.extra_attributes["Lx"])
    else:
        n_px = int(suite2p_dataset.extra_attributes["lx"])

    if "combined" in os.listdir(s2p_output_path):
        plane_paths = [p for p in os.listdir(s2p_output_path) if "plane" in p]
        meanImg = np.zeros((len(plane_paths), n_px, n_px))
        plane_paths.sort()
        for i, iplane in enumerate(plane_paths):
            s2p_ops = np.load(s2p_output_path / iplane / "ops.npy", allow_pickle=True)
            s2p_ops = s2p_ops.item()
            meanImg[i, :, :] = s2p_ops["meanImg"]
    elif "plane0" in os.listdir(s2p_output_path):
        s2p_ops = np.load(
            s2p_output_path / "plane0" / "ops.npy", allow_pickle=True
        ).item()
        meanImg = s2p_ops["meanImg"]
    else:
        FileNotFoundError("ops.npy not found in combined or plane0 directory.")

    return meanImg