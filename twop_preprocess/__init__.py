try:
    from twop_preprocess._version import __version__
except ImportError:
    # Package was not installed via pip / setuptools_scm has not written _version.py yet.
    # This can happen when running directly from the source tree without installing.
    __version__ = "unknown"

__all__ = ["__version__"]
