from tifffile import TiffFile
from pathlib import Path
from suite2p import default_ops
import os
import yaml


def parse_si_metadata(tiff_path):
    """
    Reads metadata from a Scanimage TIFF and return a dictionary with
    specified key values.

    Currently can only extract numerical data.

    Args:
        tiff_path: path to TIFF or directory containing tiffs

    Returns:
        dict: dictionary of SI parameters

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
    Generate a dictionary of ops by updating suite2p defaults and
    pipeline defaults with user specified ops.

    Args:
        user_ops: dictionary of user specified ops
        zstack: boolean, if True, load zstack ops

    Returns:
        dict: dictionary of ops

    """
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
