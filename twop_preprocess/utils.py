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
    if not tiff_path.endswith(".tif"):
        tiffs = [tiff for tiff in os.listdir(tiff_path) if tiff.endswith(".tif")]
    else:
        tiffs = [
            tiff_path,
        ]
    if tiffs:
        tiff_path = str(Path(tiff_path) / tiffs[0])
        tif = TiffFile(tiff_path)
        return tif.scanimage_metadata["FrameData"]
    else:
        return None


def load_ops(ops):
    """
    Generate a dictionary of ops by updating suite2p defaults and
    pipeline defaults with user specified ops.

    Args:
        ops: dictionary of user specified ops

    Returns:
        dict: dictionary of ops

    """
    suite2p_ops = default_ops()
    default_ops_fname = Path(__file__).parent / "default_ops.yml"
    with open(default_ops_fname, "r") as f:
        pipeline_ops = yaml.safe_load(f)
    # update suite2p defaults with pipeline defaults
    ops = dict(suite2p_ops, **pipeline_ops)
    # update with user specified config
    config_path = Path.home() / ".2p_preprocess" / "config.yml"
    if config_path.is_file():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        # update with user specified config
        ops = dict(ops, **config)
    # update with user specified ops provided at runtime
    ops = dict(ops, **ops)
    return ops
