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

class CloudMask():
    """
    Cloud mask for in-memory arrays. Can be used for any instrument, but subclasses are provided for Sentinel-2 and Landsat 8/9.

    When using this base class with a sensor independent model, the descriptors must be provided in the constructor or as an 
    argument to __call__. This is different to the subclasses for Sentinel-2 and Landsat 8/9, which have hard-coded band descriptors.

    Users are encouraged to create their own subclasses for other sensors, which can then be used in the same way as the provided subclasses.
    """

    def __init__(self,model_config,weights,descriptors=None,device='cuda',cache_scene=True,output_style='4-class',categorise=False,verbose=False):
        self.model_config = model_config
        self.model_config = yaml.load(open(self.model_config,'r'),Loader=yaml.FullLoader)
        self.weights = weights
        self.device = device
        self.cache_scene = cache_scene
        self.descriptors = descriptors
        self.model = models.load_model(self.model_config,weights=self.weights,device=self.device).eval()
        self.output_style = output_style
        self.categorise = categorise
        self.verbose = verbose

        if self.cache_scene:
            self.scene = None

    def get_scene(self,scene,resolution=None):
        """
        Expects scene to be precomputed as a numpy array.
        """
        return scene
    
    def postprocess(self,mask):
        """
        Postprocesses the raw mask output from the model.
        """

        # Use model class structure as-is
        if self.verbose:
            print('Postprocessing mask...')

        if self.output_style is None:
            pass
            
        # Clear, Thin, Thick, Shadow
        elif self.output_style == '4-class':
            if self.model_config['CLASSES']==4:
                pass
            elif self.model_config['CLASSES']==7:
                new_mask = np.zeros((4,*mask.shape[1:]))
                new_mask[0,...] = mask[:3,...].sum(axis=0) # Clear
                new_mask[1:,...] = mask[3:6,...] # Thin, Thick, Shadow
                mask = new_mask
            else:
                raise ValueError('Model config has {} classes, but output_style is set to "4-class". For this to work, the model must either be a 4- or 7-class model.'.format(self.model_config['CLASSES']))

        elif self.output_style == 'cloud-noncloud':
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
    

    def __call__(self,scene,descriptors=None,stride=None,resolution=None):
        if descriptors is None:
            descriptors = self.descriptors
        
        if descriptors is None and self.model_config.get('SEnSeIv2',False):
            raise ValueError('Descriptors must be provided for SEnSeI-enabled model, either in the constructor or as an argument to __call__')

        data = self.get_scene(scene,resolution=resolution)

        # Use stride=patch_size with warning
        if stride is None:
            warn('Stride not provided, using patch size of model as stride.')
            stride = self.model_config['PATCH_SIZE']
        sw = SlidingWindow(data,self.descriptors,stride,self.model_config['PATCH_SIZE'],verbose=self.verbose)
        mask = sw.predict(self.model)
        mask = self.postprocess(mask)
        return mask

class Sentinel2CloudMask(CloudMask):
    """
    Sentinel-2 Level-1C Cloud Mask for .SAFE format folders.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._bands = [
            {"name": "B01", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 425.0, "max_wavelength": 461.0},
            {"name": "B02", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 446.0, "max_wavelength": 542.0},
            {"name": "B03", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 537.5, "max_wavelength": 582.5},
            {"name": "B04", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 645.5, "max_wavelength": 684.5},
            {"name": "B05", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 694.0, "max_wavelength": 714.0},
            {"name": "B06", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 731.0, "max_wavelength": 749.0},
            {"name": "B07", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 767.0, "max_wavelength": 795.0},
            {"name": "B08", "resolution": 10, "band_type": "TOA Reflectance", "min_wavelength": 763.5, "max_wavelength": 904.5},
            {"name": "B8A", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 847.5, "max_wavelength": 880.5},
            {"name": "B09", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 930.5, "max_wavelength": 957.5},
            {"name": "B10", "resolution": 60, "band_type": "TOA Reflectance", "min_wavelength": 1337.0, "max_wavelength": 1413.0},
            {"name": "B11", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 1541.0, "max_wavelength": 1683.0},
            {"name": "B12", "resolution": 20, "band_type": "TOA Reflectance", "min_wavelength": 2074.0, "max_wavelength": 2314.0},
        ]   
        if self.descriptors is None:
            self.descriptors = []
            for band in self._bands:
                self.descriptors.append({
                    'band_type': band['band_type'],
                    'min_wavelength': band['min_wavelength'],
                    'max_wavelength': band['max_wavelength']
                })

    def get_scene(self,scene,resolution=10):
        """
        Parameters
        ----------
        scene: str, path to scene in .SAFE format

        Returns
        -------
        data: np.array, shape (n_bands, height, width)
        """

        data_dir = os.path.join(scene,'GRANULE')
        data_dir = os.path.join(data_dir,os.listdir(data_dir)[0],'IMG_DATA')
        if resolution is None:
            resolution = self._bands[1]['resolution']

        data = []
        with rio.open([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('B02.jp2')][0]) as src:
            b2_shape = src.shape
        if resolution == self._bands[1]['resolution']:
            b_shape = b2_shape
        else:
            factor = self._bands[1]['resolution'] / resolution
            b_shape = (int(b2_shape[0]*factor),int(b2_shape[1]*factor))

        if self.verbose:
            iter = tqdm(self._bands,desc='Reading and processing Sentinel-2 bands')
        else:
            iter = self._bands
        
        try:
            processing_baseline = int(scene.split('_')[-2][2])
        except:
            processing_baseline = 2
            print('Warning: scene processing baseline not found in scene name. Does the .SAFE folder have its original name? Assuming band offset is not needed.')

        if processing_baseline >= 4:
            offset_correction = True
        else:
            offset_correction = False

        for band in iter:
            band_path = [f for f in os.listdir(data_dir) if f.endswith(band['name']+'.jp2')][0]
            band_path = os.path.join(data_dir,band_path)
            with rio.open(band_path) as src:
                band_data = src.read(1)
            
            band_data = self._normalise_band(band_data,offset_correction)

            # Tie to shape of band B02
            if not np.all(b_shape == band_data.shape):
                band_data = transform.resize(band_data,b_shape,preserve_range=True,order=1)

            data.append(band_data)
        data = np.stack(data,axis=0)
        if self.cache_scene:
            self.scene = data
        return data
    
    def _normalise_band(self,band,offset_correction=False):
        if offset_correction:
            band = (band.astype('float32')-1000)/10_000
        else:
            band = band.astype('float32')/10_000
        return band


class Landsat89CloudMask(CloudMask):
    """
    Landsat 8/9 Level-1C Cloud Mask for folders containing the bands' .TIF files.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._bands = [
            {"name": "B1", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 435.0, "max_wavelength": 451.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B2", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 452.0, "max_wavelength": 512.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B3", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 533.5, "max_wavelength": 590.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B4", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 636.5, "max_wavelength": 673.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B5", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 851.0, "max_wavelength": 879.0, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B6", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 1566.5, "max_wavelength": 1651.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B7", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 2114.5, "max_wavelength": 2287.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B8", "resolution": 15, "band_type": "TOA Reflectance", "min_wavelength": 496.5, "max_wavelength": 683.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B9", "resolution": 30, "band_type": "TOA Reflectance", "min_wavelength": 1363.5, "max_wavelength": 1384.5, 'GAIN': 0.00002, 'OFFSET': -0.1, 'SOLAR_CORRECTION': True},
            {"name": "B10", "resolution": 30, "band_type": "TOA Normalised Brightness Temperature", "min_wavelength": 10600.0, "max_wavelength": 11190.0, 'GAIN': 0.0003342, 'OFFSET': 0.1, 'K1': 774.8853, 'K2': 1321.0789, 'MINIMUM_BT': 132.0, 'MAXIMUM_BT': 249.0, 'SOLAR_CORRECTION': False},
            {"name": "B11", "resolution": 30, "band_type": "TOA Normalised Brightness Temperature", "min_wavelength": 11500.0, "max_wavelength": 12510.0, 'GAIN': 0.0003342, 'OFFSET': 0.1, 'K1': 480.8883, 'K2': 1201.1442, 'MINIMUM_BT': 127.0, 'MAXIMUM_BT': 239.0, 'SOLAR_CORRECTION': False},
        ]   
        if self.descriptors is None:
            self.descriptors = []
            for band in self._bands:
                self.descriptors.append({
                    'band_type': band['band_type'],
                    'min_wavelength': band['min_wavelength'],
                    'max_wavelength': band['max_wavelength']
                })


    def get_scene(self,scene,resolution=30):
        """
        Parameters
        ----------
        scene: str, path to scene in .SAFE format

        Returns
        -------
        data: np.array, shape (n_bands, height, width)
        """

        data_dir = scene

        # Need to do solar angle correction from metadata
        metadata_path = os.path.join(data_dir,[f for f in os.listdir(data_dir) if f.endswith('_MTL.txt')][0])
        
        metadata = self.load_scene_metadata(metadata_path)

        data = []

        # Get B1 to tie shape of other bands to it
        with rio.open([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.endswith('B1.TIF')][0]) as src:
            b1_shape = src.shape
        if resolution == self._bands[0]['resolution']:
            b_shape = b1_shape
        else:
            factor = self._bands[0]['resolution'] / resolution
            b_shape = (int(b1_shape[0]*factor),int(b1_shape[1]*factor))
        
        if self.verbose:
            iter = tqdm(self._bands,desc='Reading and processing Landsat 8/9 bands')
        else:
            iter = self._bands

        for band in iter:
            band_path = [f for f in os.listdir(data_dir) if f.endswith(band['name']+'.TIF')][0]
            band_path = os.path.join(data_dir,band_path)
            with rio.open(band_path) as src:
                band_data = src.read(1)
                
            # Tie to shape of band B1
            if not np.all(b_shape == band_data.shape):
                band_data = transform.resize(band_data,b_shape,preserve_range=True,order=1)
            band_data = self._normalise_band(band_data,band,metadata)
            data.append(band_data)
        data = np.stack(data,axis=0)
        if self.cache_scene:
            self.scene = data
        return data

    def load_scene_metadata(self, path):
        # Taken from eo4ai/loaders.py
        """Loads metadata values from a given path to Landsat metadata file.

        Parameters
        ----------
        path : str
            Path to Landsat 8 metadata file.

        Returns
        -------
        config : dict
            Dictionary containing all values from Landsat metadata file
            (non-hierarchical).
        """
        with open(path) as f:
            config = {
                entry[0]: entry[1]
                for entry in map(lambda l: "".join(l.split()).split('='), f)
                if len(entry) == 2
            }
        for k, v in config.items():
            try:
                config[k] = float(v)
            except ValueError:
                continue
        return config

    def _normalise_band(self, band, band_metadata, scene_metadata, nodata_as=None):
        """Normalises any Landsat 8 band into TOA units

        Parameters
        ----------
        band : np.ndarray
            Array with Digital Number pixel values from Landsat 8.
        band_id : str
            Identifier for band being normalised.
        scene_metadata : dict
            Scene-specific Landsat metadata values.
        nodata_as : float, optional
            Used to set nodata as given value, left as is if unspecified
            (default is None)


        Returns
        -------
        band : np.ndarray
            Normalised band in TOA units.
        """
        # Get a shortcut for the band's metadata
        bm = band_metadata

        gain = bm['GAIN']
        offset = bm['OFFSET']

        nodata = band == 0
        band = band * gain + offset
        if bm['band_type'] == 'TOA Normalised Brightness Temperature':
            band = (bm['K2'] / np.log(bm['K1'] / band + 1))
            band = (band - bm['MINIMUM_BT']) \
                / (bm['MAXIMUM_BT'] - bm['MINIMUM_BT'])

        if bm.get('SOLAR_CORRECTION', False):
            band /= math.sin(
                        float(scene_metadata['SUN_ELEVATION'])
                        * math.pi / 180
                        )

        if nodata_as is not None:
            band[nodata] = nodata_as
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
    parser.add_argument('-s', '--stride', default=256, type=int, help='Stride to use for inference. If not provided, will use patch size of model.')
    parser.add_argument('-r', '--resolution', default=None, type=float, help='Resolution of output mask (metres). If not provided, will use 10m for Sentinel-2 and 30m for Landsat 8/9.')
    parser.add_argument('-c', '--categorise', action='store_true', help='Categorise mask into classes, rather than softmax confidences.')
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
        cm = Sentinel2CloudMask(args.model_config,args.weights,device=args.device,output_style=args.output_style,categorise=args.categorise,verbose=args.verbose)
        # Get profile, to be used for output
        band_dir = os.path.join(args.scene,'GRANULE')
        band_dir = os.path.join(band_dir,os.listdir(band_dir)[0],'IMG_DATA')
        with rio.open(os.path.join(band_dir,[f for f in os.listdir(band_dir) if f.endswith('B02.jp2')][0])) as src:
            profile = src.profile
            
    elif args.instrument.lower() == 'landsat89':
        if args.resolution is None:
            args.resolution = 30
        cm = Landsat89CloudMask(args.model_config,args.weights,device=args.device,output_style=args.output_style,categorise=args.categorise,verbose=args.verbose)
        # Get profile, to be used for output
        with rio.open(os.path.join(args.scene,[f for f in os.listdir(args.scene) if f.endswith('B1.TIF')][0])) as src:
            profile = src.profile

    else:
        raise ValueError('Instrument must be one of Sentinel2 or Landsat89.')

    mask = cm(args.scene,stride=args.stride,resolution=args.resolution)

    write_mask(mask,profile,args)

    return None

if __name__ == '__main__':
    main()