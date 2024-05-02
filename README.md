# SEnSeIv2
Sensor Independent Cloud and Shadow Masking with Partial Labels and Multimodal Inputs


## Installation

Before installing SEnSeIv2 with either method below, make sure you have a working installation of PyTorch and torchvision (with CUDA drivers, if you want to use a GPU). [More info on installing pytorch here](https://pytorch.org/get-started/locally/). This is _not included_ in the requirements.txt file. 

#### Install with pip


To install SEnSeIv2 as both a python package and a command line tool:

```
pip install senseiv2
```

This will download the default model weights from the [SEnSeIv2 HuggingFace model repo](https://huggingface.co/aliFrancis/SEnSeIv2). Other model weights used in the paper can also be found here, if you are interested.

#### Source code

If you would like to train models yourself, or use or adapt the code, you can clone this repository.

```
git clone git@github.com:aliFrancis/SEnSeIv2.git
cd ./SEnSeIv2
python setup.py install
```

## Basic Usage

#### Command line interface

As an example, you can produce a cloud mask (with classes _clear_, _thin_, _thick_ and _cloud shadow_) for a Sentinel-2 scene

```
senseiv2 -v sentinel2 <path/to/S2-scene.SAFE> <path/to/output.tif>
```
Or, for Landsat 8 or 9:
```
senseiv2 -v landsat89 <path/to/landsat89-scene> <path/to/output.tif>
```

To see all options for the command line tool, which, for instance, allow you to control parameters such as the class structure of the mask, or its resolution, you can use:

```
senseiv2 --help
```

#### Importing in python

You can use the cloud masks within python, if you are doing your own data preprocessing, or want to customise things in other ways. A typical use-case might begin with:

```python
from senseiv2.inference import CloudMask
from senseiv2.utils import get_model_files

scene = ... # Some numpy array representing a satellite image
descriptors = [
  {...},    # See samples/ for examples of descriptor dictionaries
  {...}
]

# Pick pre-trained model from https://huggingface.co/aliFrancis/SEnSeIv2
model_name = 'SEnSeIv2-SegFormerB2-alldata-ambiguous'
config, weights = get_model_files(model_name)

# Lots of options in the kwargs for different settings
cm = CloudMask(config, weights, descriptors=descriptors,verbose=True)

mask = cm(scene)
```

#### Advanced uses (model training etc.)

It is not easy to replicate precisely the training strategy used here, because it is not possible to redistribute all the datasets used. However, the train.py script is included, along with some sample data in samples/ to get you started. All the data used has been preprocessed into a shared format using the [eo4ai tool](https://github.com/ESA-PhiLab/eo4ai).

After collecting some of the datasets, you can use the training script with a config file (with some modifications) from [the HuggingFace repo](https://huggingface.co/aliFrancis/SEnSeIv2):

```
python train.py path/to/config.yaml
```
