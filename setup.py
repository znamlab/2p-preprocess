from setuptools import setup, find_packages

setup(
    name="2p-preprocess",
    version="0.1.1",
    packages=find_packages(
        where="src", include=["neuropil", "preprocess2p", "plotting_utils"]
    ),
    package_dir={
        "": "src",
    },
    install_requires=[
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git@dev",
        "matplotlib",
        "jupyter",
        "defopt",
        "more_itertools",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "neuropil = neuropil.neuropil:main",
            "preprocess2p = preprocess2p.preprocess2p:entry_point",
        ],
    },
)
