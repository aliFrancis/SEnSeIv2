# setup.py for SEnSeIv2

from setuptools import setup, find_packages
import os

REQUIRED = [
    "huggingface-hub >= 0.17",
    "numpy",
    "matplotlib >= 3.6.2",
    "pyyaml >= 6.0",
    "rasterio >= 1.3.4",
    "scikit-image >= 0.19.3",
    "segmentation-models-pytorch == 0.3.2",
    "six",
    "torchmetrics >= 1.1",
    "transformers >= 4.34.0",
    "wandb >= 0.15.12"
]



def package_files(directory):
    paths = []
    for (r, d, f) in os.walk(directory):
        for filename in f:
            paths.append(os.path.join('..', r, filename))
    return paths

# extra_files = package_files('hf_models')


setup(
    name='senseiv2',
    packages=find_packages(exclude=['hf_models/*']),
    # package_data={'': extra_files},
    entry_points={
        'console_scripts': [
            'senseiv2 = senseiv2.inference:main'
        ]
    },
    version='0.1',
    description='SEnSeIv2',
    author='Alistair Francis',
    install_requires=REQUIRED,
)
