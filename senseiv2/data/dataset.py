import json
import numpy as np
import os
import torch
import warnings


class SEnSeIv2Dataset(torch.utils.data.Dataset):
    """
    Customisable generator for use with keras' model.fit() method. Does not hold
    dataset in memory, making it ideal for large quantities of data, but it is
    recommended to use as fast a hard drive as possible.

    Dataset should be saved on disk using format from https://github.com/ESA-PhiLab/eo4ai

    Attributes
    ----------
    dirs : str
        Path to root directory of dataset. Dataset will recursively search for valid
        subdirectories (see self.parse_dirs() for information on the criteria used).
    patch_size : int
        Size of returned images and masks
    transform : list, optional
        Functions like those defined in sensei.data.transformations, used to
          augment data. Recommended to at least use sensei.data.transformations.Base
    band_selection : str / int / tuple / list, optional
        How to select the spectral bands of the data:
          - str: "all" fixed selection of all bands.
          - int: Random selection of that number of bands.
          - tuple: Random selection of bands, with some number between values in tuple.
          - list: Fixed selection of those bands at indices defined in list.
    allowed_band_types: list, optional
        List of allowed band types, e.g. 'TOA Reflectance' etc.
    output_metadata : bool, optional
        Useful for checking dataset integrity/debugging, allows output of each example's
          metadata in the batch
    output_descriptor_dicts : bool, optional
        Required for use with SEnSeI, adds descriptors to batch
    repeated : int, optional
        Can be used to simply repeat the dataset by a given number of times, which
          gets around keras' model.fit() running out of data.
    return_paths : bool, optional
        If True, returns the path of each example in the batch. Useful for debugging.
    im_filename : str, optional
        Name of image file in each subdirectory. Default is 'image.npy'.
    mask_filename : str, optional
        Name of mask file in each subdirectory. Default is 'mask.npy'. Particularly useful
        when different kinds of mask are used (ambiguous vs. one-hot format).
    metadata_filename : str, optional
        Name of metadata file in each subdirectory. Default is 'metadata.json'. Often used
        together with mask_filename to indicate the type of mask used.
    """

    def __init__(self, dirs, patch_size,
                    transform=None, band_selection=None,
                    allowed_band_types=None, output_metadata=False,
                    output_descriptor_dicts=True, repeated=False, 
                    return_paths=False,im_filename='image.npy',
                    mask_filename='mask.npy',metadata_filename='metadata.json'):
        self.dirs = dirs
        self.repeated = repeated
        self.patch_size = patch_size
        self.transform = transform
        self.band_selection = band_selection
        self.allowed_band_types = allowed_band_types
        self.output_descriptor_dicts = output_descriptor_dicts
        self.output_metadata = output_metadata
        self.return_paths = return_paths
        self.im_filename = im_filename
        self.mask_filename = mask_filename
        self.metadata_filename = metadata_filename


        self.paths = self.parse_dirs()  # returns full paths for annotation folders
        if self.repeated:
            self.paths = self.paths*self.repeated
        self.N_samples = len(self.paths)

        print('Dataset created with {} samples'.format(self.N_samples))

    def __len__(self):
        return self.N_samples

    def __getitem__(self, idx):
        self.band_policy = self._get_band_policy()
        path = self.paths[idx]
        if self.return_paths:
            im, mask, metadata, path = self._read_and_transform(path)
            if self.output_descriptor_dicts:
                return_tuple = ((im,metadata['bands']), mask, path)
            else:
                return_tuple = (im,mask,path)
        else:
            im, mask, metadata = self._read_and_transform(path)
            if self.output_descriptor_dicts:
                return_tuple = ((im,metadata['bands']), mask)
            else:
                return_tuple = (im,mask)
        return return_tuple

    def _read_and_transform(self, paths):
        if self.return_paths:
            im,mask,metadata,path = self._read(paths)
        else:
            im,mask,metadata = self._read(paths)
        im,metadata = self._select_bands(im,metadata)
        if self.transform is not None:
            for transform in self.transform:
                im,mask,metadata = transform(im,mask,metadata)
        
        if self.return_paths:
            return im, mask, metadata, path
        else:
            return im, mask, metadata

    def parse_dirs(self):
        """
        Look for all valid subdirectories in self.dirs.

        Returns
        -------
        valid_subdirs : list
            Paths to all subdirectories containing:
              - image.npy
              - mask.npy
              - metadata.json
        """
        valid_subdirs = []
        if isinstance(self.dirs, str):
            self.dirs = [self.dirs]
        for dir in self.dirs:
            for root, dirs, paths in os.walk(dir):
                valid_subdirs += [
                            os.path.join(root, dir) for dir in dirs
                            if  os.path.isfile(os.path.join(root, dir, self.im_filename))
                            and os.path.isfile(os.path.join(root, dir, self.mask_filename))
                            and os.path.isfile(os.path.join(root, dir, self.metadata_filename))
                            ]
        return valid_subdirs

    def _read(self,path):
        filenames = [self.im_filename,self.mask_filename,self.metadata_filename]

        image_file, mask_file, metadata_file = [os.path.join(path,fname) for fname in filenames]

        # Read with mmap_mode='r' to avoid loading into memory, which is slower
        im = np.load(image_file,mmap_mode='r')

        mask = np.load(mask_file,mmap_mode='r')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        metadata['bands'] = metadata['bands']

        im,metadata = self.disallow_band_types(im,metadata)
        if self.return_paths:
            return im, mask, metadata, path
        else:
            return im, mask, metadata

    def disallow_band_types(self,im,metadata):
        if self.allowed_band_types is not None:
            allowed_band_indices = [i for i,band in enumerate(metadata['bands']) if band['band_type'] in self.allowed_band_types]
            im = im[...,allowed_band_indices]
            metadata['bands'] = [metadata['bands'][i] for i in allowed_band_indices]
        return im,metadata

    def _select_bands(self,im,metadata):
        padding = 0

        if self.band_policy is None or self.band_policy == 'all':
            return im,metadata
        elif isinstance(self.band_policy,list):
            band_indices = self.band_policy
        elif isinstance(self.band_policy,int):
            if self.band_policy <= len(metadata['bands']):
                band_indices = np.random.choice(list(range(len(metadata['bands']))), self.band_policy, replace=False)

            else:
                band_indices = list(range(len(metadata['bands'])))
                padding = self.band_policy - len(metadata['bands'])
        else:
            print('Band policy:  {}  not recognised.'.format(self.band_policy))

        im = im[...,band_indices]
        metadata['bands'] = [metadata['bands'][b] for b in band_indices]

        if padding:
            im = np.concatenate([im, -0.5*np.ones((*im.shape[:-1], padding))], axis=-1)
            metadata['bands'] += [{'band_type':'fill'}] * padding
        return im,metadata

    def _get_band_policy(self):
        if isinstance(self.band_selection,tuple):
            return np.random.randint(*self.band_selection)
        elif isinstance(self.band_selection,str):
            return self.band_selection
        elif isinstance(self.band_selection,list):
            return self.band_selection
        else:
            return None
    
    def collate_fn(self,samples):
        if self.return_paths:
            ins, masks, paths = zip(*samples)
        else:
            ins, masks = zip(*samples)

        if self.output_descriptor_dicts:
            images, descriptors = zip(*ins)
        else:
            images = ins
        
        channels = max([im.shape[-1] for im in images])

        # pad images to max channels
        images = [np.concatenate([im, -0.5*np.ones((*im.shape[:-1], channels-im.shape[-1]))], axis=-1) for im in images]

        # pad metadata bands to max channels
        if self.output_descriptor_dicts:
            descriptors = [d + [{'band_type':'fill'}] * (channels-len(d)) for d in descriptors]
            for D in descriptors:
                for d in D:
                    for k,v in d.items():
                        if isinstance(v,np.ndarray) or isinstance(v,list):
                            d[k] = torch.Tensor(v)
        # stack images and masks (descriptors stay as list of dicts)
        images = torch.Tensor(np.moveaxis(np.stack(images),-1,1))
        
        masks = torch.Tensor(np.moveaxis(np.stack(masks),-1,1))

        return_dict = {}
        if self.output_descriptor_dicts:
            return_dict['inputs'] = (images,descriptors)
        else:
            return_dict['inputs'] = images
        return_dict['labels'] = masks

        if self.return_paths:
            return_dict['paths'] = paths

        return return_dict
            
