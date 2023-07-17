from setuptools import setup, find_packages

setup(
    name="2p-preprocess",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git",
        "matplotlib",
        "jupyter",
        "more_itertools",
        "scikit-image",
        "tqdm",
        "Click",
    ],
    entry_points="""
        [console_scripts]
        2p=twop_preprocess.cli:cli
    """,
)
