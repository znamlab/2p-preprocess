from tifffile import TiffFile
from pathlib import Path
import os
import yaml


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
