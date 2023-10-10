# setup.py for SEnSeIv2

import huggingface_hub as hf_hub
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

# Get initial model
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='config.yaml',subfolder='full-models/SEnSeIv2-SegFormerB2-alldata-ambiguous', local_dir='./hf_models/full-models/SEnSeIv2-SegFormerB2-alldata-ambiguous')
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='weights.pt',subfolder='full-models/SEnSeIv2-SegFormerB2-alldata-ambiguous', local_dir='./hf_models/full-models/SEnSeIv2-SegFormerB2-alldata-ambiguous')
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='senseiv2-small.yaml',subfolder='sensei-configs', local_dir='./hf_models/sensei-configs')
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='senseiv2-medium.yaml',subfolder='sensei-configs', local_dir='./hf_models/sensei-configs')
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='senseiv2-big.yaml',subfolder='sensei-configs', local_dir='./hf_models/sensei-configs')
# hf_hub.hf_hub_download(repo_id='aliFrancis/SEnSeIv2',filename='senseiv1.yaml',subfolder='sensei-configs', local_dir='./hf_models/sensei-configs')


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
