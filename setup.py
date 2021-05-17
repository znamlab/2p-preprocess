from setuptools import setup, find_packages

setup(
    name='2p-preprocess',
    version='0.1.0',
    packages = find_packages(where="src", include=["neuropil", "preprocess2p"]),
    package_dir={
        '': 'src',
    },
    install_requires=[
        'flexiznam @ git+ssh://git@github.com/znamlab/flexiznam.git',
        'matplotlib',
        'jupyter',
        'defopt'
    ],
    entry_points={
        'console_scripts': [
            'neuropil = neuropil.neuropil:main',
            'preprocess2p = preprocess2p.preprocess2p:entry_point'
        ],
    }
    )
