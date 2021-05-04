from setuptools import setup, find_packages

setup(
    name='2p-preprocess',
    version='0.1.0',
    packages = find_packages(where="src", include=["neuropil"]),
    package_dir={
        '': 'src',
    },
    entry_points={
        'console_scripts': [
            'neuropil = neuropil.neuropil:main',
        ],
    }
    )
