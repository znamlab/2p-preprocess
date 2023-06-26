from tifffile import TiffFile
from pathlib import Path
import os


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