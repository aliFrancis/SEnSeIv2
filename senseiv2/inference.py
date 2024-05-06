import math
import numpy as np
import os
import rasterio as rio
from skimage import transform
from tqdm import tqdm
from warnings import warn
import yaml

from senseiv2.data.sliding_window import SlidingWindow
from senseiv2 import models
from senseiv2.utils import get_model_files
from senseiv2.constants import SENTINEL2_BANDS, LANDSAT89_BANDS, SENTINEL2_DESCRIPTORS, LANDSAT89_DESCRIPTORS

class CloudMask():
    """
    Cloud mask for in-memory arrays. Can be used for any instrument, but subclasses are provided for Sentinel-2 and Landsat 8/9.

    When using this base class with a sensor independent model, the descriptors must be provided in the constructor or as an 
    argument to __call__. This is different to the subclasses for Sentinel-2 and Landsat 8/9, which have hard-coded band descriptors.

    Users are encouraged to create their own subclasses for other sensors, which can then be used in the same way as the provided subclasses.
    """

    def __init__(self,model_config,weights,device='cuda',cache_scene=True,output_style='4-class',categorise=False,batch_size=1,verbose=False):
        self.model_config = model_config
        self.model_config = yaml.load(open(self.model_config,'r'),Loader=yaml.FullLoader)
        self.weights = weights
        self.device = device
        self.cache_scene = cache_scene
        self.model = models.load_model(self.model_config,weights=self.weights,device=self.device).eval()
        self.output_style = output_style
        self.categorise = categorise
        self.batch_size = batch_size
        self.verbose = verbose

        if self.cache_scene:
            self.scene = None
    
    def postprocess(self,mask):
        """
        Postprocesses the raw mask output from the model.
        """

        # Use model class structure as-is
        if self.output_style is None:
            pass
            
        # Clear, Thick, Thin, Shadow
        elif self.output_style == '4-class':
            if self.verbose:
                print('Postprocessing mask with classes: Clear/Thick/Thin/Shadow')
            if self.model_config['CLASSES']==4:
                # Assumes classes are already in the correct order (above)
                pass
            elif self.model_config['CLASSES']==7:
                new_mask = np.zeros((4,*mask.shape[1:]))
                new_mask[0,...] = mask[:3,...].sum(axis=0) # Clear
                new_mask[1,...] = mask[4,...] # Thick (ordering in 4-class mode is thick,thin for compatibility)
                new_mask[2,...] = mask[3,...] # Thin
                new_mask[3,...] = mask[5,...] # Shadow
                mask = new_mask
            else:
                raise ValueError('Model config has {} classes, but output_style is set to "4-class". For this to work, the model must either be a 4- or 7-class model.'.format(self.model_config['CLASSES']))

        elif self.output_style == 'cloud-noncloud':
            if self.verbose:
                print('Postprocessing mask with classes: Non-cloud/Cloud')
            if self.model_config['CLASSES']==2:
                pass
            if self.model_config['CLASSES']==4:
                new_mask = np.zeros((2,*mask.shape[1:]))
                new_mask[1,...] = mask[1:3,...].sum(axis=0) # Cloud
                new_mask[0,...] = 1-new_mask[1,...] # Non-cloud
                mask = new_mask
            elif self.model_config['CLASSES']==7:
                new_mask = np.zeros((2,*mask.shape[1:]))
                new_mask[1,...] = mask[3:5,...].sum(axis=0) # Cloud
                new_mask[0,...] = 1-new_mask[1,...] # Non-cloud
                mask = new_mask
            else:
                raise ValueError('Model config has {} classes, but output_style is set to "cloud-noncloud". For this to work, the model must either be a 2-, 4- or 7-class model.'.format(self.model_config['CLASSES']))

        elif self.output_style == 'valid-invalid':
            if self.verbose:
                print('Postprocessing mask with classes: Valid/Invalid(cloud+shadow)')
            if self.model_config['CLASSES']==2:
                pass
            if self.model_config['CLASSES']==4:
                new_mask = np.zeros((2,*mask.shape[1:]))
                new_mask[1,...] = mask[1:4,...].sum(axis=0) # Invalid
                new_mask[0,...] = 1-new_mask[1,...] # Valid
                mask = new_mask
            elif self.model_config['CLASSES']==7:
                new_mask = np.zeros((2,*mask.shape[1:]))
                new_mask[1,...] = mask[3:6,...].sum(axis=0) # Invalid
                new_mask[0,...] = 1-new_mask[1,...] # Valid
                mask = new_mask
            else:
                raise ValueError('Model config has {} classes, but output_style is set to "cloud-noncloud". For this to work, the model must either be a 2-, 4- or 7-class model.'.format(self.model_config['CLASSES']))

        else:
            raise ValueError('output_style must be one of None (whatever the model outputs), "4-class", "cloud-noncloud" or "valid-invalid".')

        if self.categorise:
            mask = np.argmax(mask,axis=0)
        
        return mask
    

    def __call__(self,data,descriptors=None,stride=None,resolution=None):
        if descriptors and not self.model_config.get('SEnSeIv2',False):
            warn('Descriptors provided, but model is not SEnSeI-enabled. Descriptors will be ignored.')
        
        if descriptors is None and self.model_config.get('SEnSeIv2',False):
            raise ValueError('Descriptors must be provided for SEnSeI-enabled model, either in the constructor or as an argument to __call__')

        # Use stride=patch_size with warning
        if stride is None:
            warn('Stride not provided, using patch size of model as stride.')
            stride = self.model_config['PATCH_SIZE']
        sw = SlidingWindow(
            data,
            descriptors,
            stride,
            self.model_config['PATCH_SIZE'],
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        mask = sw.predict(self.model)
        mask = self.postprocess(mask)
        return mask

def sentinel2_loader(scene,resolution=10,verbose=False):
    # Simple loader function for a .SAFE folder
    data_dir = os.path.join(scene,'GRANULE')
    data_dir = os.path.join(data_dir,os.listdir(data_dir)[0],'IMG_DATA')
    if resolution is None:
        resolution = 10

    bands = SENTINEL2_BANDS
    data = []
    with rio.open([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('B02.jp2')][0]) as src:
        b2_shape = src.shape
    if resolution == bands[1]['resolution']:
        b_shape = b2_shape
    else:
        factor = bands[1]['resolution'] / resolution
        b_shape = (int(b2_shape[0]*factor),int(b2_shape[1]*factor))

    if verbose:
        print('Loading bands...')
        iterator = tqdm(bands)
    else:
        iterator = bands
    for band in iterator:
        band_path = [f for f in os.listdir(data_dir) if f.endswith(band['name']+'.jp2')][0]
        band_path = os.path.join(data_dir,band_path)
        with rio.open(band_path) as src:
            band_data = src.read(1)
        
        # Tie to shape of band B02
        if not np.all(b_shape == band_data.shape):
            band_data = transform.resize(band_data,b_shape,preserve_range=True,order=1)

        data.append(band_data)

    data = np.stack(data,axis=0)

    # Normalize
    try:
        processing_baseline = int(scene.split('_')[-2][2])
    except:
        processing_baseline = 2
        print('Warning: scene processing baseline not found in scene name. Does the .SAFE folder have its original name? Assuming band offset is not needed.')

    if processing_baseline >= 4:
        data = (data.astype('float32')-1000)/10_000
    else:
        data = data.astype('float32')/10_000

    return data

def landsat89_loader(scene,resolution=30,verbose=False):
    # Simple loader function for a folder containing the bands' .TIF files
    bands = LANDSAT89_BANDS
    data = []
    with rio.open([os.path.join(scene,f) for f in os.listdir(scene) if f.endswith(f'{bands[0]["name"]}.TIF')][0]) as src:
        b1_shape = src.shape
    if resolution == bands[0]['resolution']:
        b_shape = b1_shape
    else:
        factor = bands[1]['resolution'] / resolution
        b_shape = (int(b1_shape[0]*factor),int(b1_shape[1]*factor))

    if verbose:
        print('Loading bands...')
        iterator = tqdm(bands)
    else:
        iterator = bands
    for band in iterator:
        band_path = [f for f in os.listdir(scene) if f.endswith(band['name']+'.TIF')][0]
        band_path = os.path.join(scene,band_path)
        with rio.open(band_path) as src:
            band_data = src.read(1)
        
        # Tie to shape of band B1
        if not np.all(b_shape == band_data.shape):
            band_data = transform.resize(band_data,b_shape,preserve_range=True,order=1)

        band_data = normalise_landsat89_band(band_data,band)
        data.append(band_data)

    data = np.stack(data,axis=0)
    return data

def normalise_landsat89_band(band,band_metadata):
    if band_metadata.get('SOLAR_CORRECTION',False):
        band = band*band_metadata['GAIN']+band_metadata['OFFSET']
    else:
        band = band*band_metadata['GAIN']+band_metadata['OFFSET']
        band = band*band_metadata['K2']+band_metadata['K1']
        band = band.clip(band_metadata['MINIMUM_BT'],band_metadata['MAXIMUM_BT'])
        band = (band-band_metadata['MINIMUM_BT'])/(band_metadata['MAXIMUM_BT']-band_metadata['MINIMUM_BT'])
    return band


def write_mask(mask,profile,args):
    if args.verbose:
        print('Writing mask to {}'.format(args.output))
    # update profile for output
    profile.update(driver='GTiff')
    profile.update(compress='lzw')
    if args.categorise:
        profile.update(dtype=rio.uint8)
        profile.update(count=1)
        profile.update(height=mask.shape[0])
        profile.update(width=mask.shape[1])
    else:
        profile.update(dtype=rio.float32)
        profile.update(count=mask.shape[0])
        profile.update(height=mask.shape[1])
        profile.update(width=mask.shape[2])
    profile['transform'] = rio.Affine(
        np.sign(profile['transform'].a) * args.resolution,
        profile['transform'].b,
        profile['transform'].c,
        profile['transform'].d,
        np.sign(profile['transform'].e) * args.resolution,
        profile['transform'].f
    )   
    with rio.open(args.output, 'w', **profile) as dst:
        if args.categorise:
            dst.write(mask.astype(rio.uint8), 1)
        else:
            for i in range(mask.shape[0]):
                dst.write(mask[i,...].astype(rio.float32), i+1)
    return None

def main():

    import argparse

    parser = argparse.ArgumentParser(description='Cloud mask a scene.')
    parser.add_argument('instrument', type=str, help='Instrument (Sentinel2 or Landsat89)')
    parser.add_argument('scene', type=str, help='Path to scene folder. (.SAFE for Sentinel-2, folder containing bands\' .TIF files for Landsat 8/9)')
    parser.add_argument('output', type=str, help='Path to output mask file. (.TIF)')
    parser.add_argument('-m', '--model', default=None, type=str, help='Name of model (see https://huggingface.co/aliFrancis/SEnSeIv2)')
    parser.add_argument('-d', '--device', default='cuda', type=str, help='Torch device to run inference on. Could be e.g. `cpu\' or `cuda:0\'.')
    parser.add_argument('-s', '--stride', default=512, type=int, help='Stride to use for inference. If not provided, will use patch size of model.')
    parser.add_argument('-r', '--resolution', default=None, type=float, help='Resolution of output mask (metres). If not provided, will use 10m for Sentinel-2 and 30m for Landsat 8/9.')
    parser.add_argument('-c', '--categorise', action='store_true', help='Categorise mask into classes, rather than softmax confidences.')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='How many patches to send to GPU in each batch. Can speed up processing when larger GPU is available. Default is 1.')
    parser.add_argument(
                        '-o', '--output_style', default='4-class', type=str, help='Output style of mask. One of None (whatever the model outputs,' 
                        'could include land/water/snow which are not so reliable - use with caution), "4-class", "cloud-noncloud" or "valid-invalid".'
                        )
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode.')
    args = parser.parse_args()


    # Get model and config as absolute paths
    if args.model is None:
        args.model_config, args.weights = get_model_files('SEnSeIv2-SegFormerB2-alldata-ambiguous')
    else:
        args.model_config, args.weights = get_model_files(args.model)

    if args.instrument.lower() == 'sentinel2':
        if args.resolution is None:
            args.resolution = 10
        descriptors = SENTINEL2_DESCRIPTORS
        # Get profile, to be used for output
        band_dir = os.path.join(args.scene,'GRANULE')
        assert os.path.exists(band_dir), 'Scene folder does not contain GRANULE folder. Is this a standard Sentinel-2 .SAFE folder?'
        band_dir = os.path.join(band_dir,os.listdir(band_dir)[0],'IMG_DATA')
        with rio.open(os.path.join(band_dir,[f for f in os.listdir(band_dir) if f.endswith('B02.jp2')][0])) as src:
            profile = src.profile
        data = sentinel2_loader(args.scene,resolution=args.resolution,verbose=args.verbose)

    elif args.instrument.lower() == 'landsat89':
        if args.resolution is None:
            args.resolution = 30
        descriptors = LANDSAT89_DESCRIPTORS
        # Get profile, to be used for output
        with rio.open(os.path.join(args.scene,[f for f in os.listdir(args.scene) if f.endswith('B1.TIF')][0])) as src:
            profile = src.profile
        data = landsat89_loader(args.scene,resolution=args.resolution,verbose=args.verbose)
        
    else:
        raise ValueError('Instrument must be one of Sentinel2 or Landsat89.')
    
    cm = CloudMask(args.model_config,args.weights,device=args.device,categorise=args.categorise,batch_size=args.batch_size,output_style=args.output_style,verbose=args.verbose)
    mask = cm(data,descriptors=descriptors,stride=args.stride)
    write_mask(mask,profile,args)

    return None

if __name__ == '__main__':
    main()