"""
Calcium processing module for 2p-preprocess.

This package contains functions for extracting fluorescence traces (using Suite2p),
calculating dF/F, deconvolving spikes, splitting recordings, and related utilities.
"""

# Import the main session processing function to make it available
# at the package level, e.g., `from twop_preprocess.calcium import extract_session`
from .calcium import extract_session


__all__ = ["extract_session"]
