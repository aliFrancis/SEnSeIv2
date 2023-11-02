# setup.py for SEnSeIv2

from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()



def package_files(directory):
    paths = []
    for (r, d, f) in os.walk(directory):
        for filename in f:
            paths.append(os.path.join('..', r, filename))
    return paths

extra_files = package_files('hf_models')


setup(
    name='senseiv2',
    packages=find_packages(),
    package_data={'': extra_files},
    entry_points={
        'console_scripts': [
            'senseiv2 = senseiv2.inference:main'
        ]
    },
    version='0.1',
    description='SEnSeIv2',
    author='Alistair Francis',
    install_requires=required,
)
