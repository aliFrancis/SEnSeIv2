import numpy as np
import torch
from tqdm import tqdm

class SlidingWindow():
    """
    Sliding window for inference on large(r) images. Assumes the image has been preprocessed and ready for inference. An instance of
    this class relates to one image. SlidingWindow.predict is then called with a specific model to get the predictions for that image.
    Multiple models can be used on the same image without having to preprocess it multiple times, with different .predict calls.

    Attributes:
        bands (np.ndarray): Array of shape (C, Y, X) containing the bands of the image.
        descriptors (list): List of descriptors to be passed to the model. If None, the model is assumed to be non-sensei-enabled.
        stride (int): Stride of the sliding window. (Should be the same as or smaller than patch_size).
        patch_size (int): Size of the patches.
        batch_size (int): Batch size used for inference.
    """

    def __init__(self, bands, descriptors, stride, patch_size, batch_size=4,verbose=False):
        self.bands = bands
        self.Y, self.X = bands.shape[1:3]
        self.stride = stride
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.descriptors = descriptors
        if self.descriptors is not None:
            assert len(self.descriptors) == self.bands.shape[0], 'Number of descriptors must match number of bands.'
        self.verbose = verbose
        self.offsets = self.calculate_offsets()
        self.coords = self.get_coords()
        self.bordered_bands = self.add_borders(bands)

    def calculate_offsets(self):
        # Calculate offsets to centre patch
        steps_Y = (self.Y - self.stride) // self.stride + 1
        steps_X = (self.X - self.stride) // self.stride + 1
        remainder_y = self.patch_size + steps_Y*self.stride - self.Y
        remainder_x = self.patch_size + steps_X*self.stride - self.X
        offset_y = remainder_y // 2
        offset_x = remainder_x // 2

        if self.Y // 2:
            # If odd, add one extra pixel to bottom
            offsets_y = (offset_y, offset_y + 1)
        else:
            offsets_y = (offset_y, offset_y)
        
        if self.X // 2:
            # If odd, add one extra pixel to right
            offsets_x = (offset_x, offset_x + 1)
        else:
            offsets_x = (offset_x, offset_x)

        return offsets_y, offsets_x
    
    def add_borders(self, bands):
        # Add reflected margin to image (size of offset)
        offsets_y, offsets_x = self.offsets

        return np.pad(bands, ((0,0), offsets_y, offsets_x), 'reflect')

    def get_coords(self):
        # Get top-left coordinates of each patch
        coords = []
        for y in range(0, self.Y, self.stride):
            for x in range(0, self.X, self.stride):
                coords.append((y, x))
        return coords

    def get_patches(self,coords):
        # Get patches from coordinates
        patches = []
        for y, x in coords:
            patches.append(self.bordered_bands[:, y:y+self.patch_size, x:x+self.patch_size])
        return patches

    def add_patches(self, mask, preds, coords):
        # Add predictions to mask
        for pred, (y, x) in zip(preds, coords):
            mask[:, y:y+self.patch_size, x:x+self.patch_size] += pred.detach().cpu().numpy()
        return mask

    def get_class_num(self, model):
        # Get number of classes from model using dummy input
        if self.descriptors is None:
            dummy = torch.zeros((self.batch_size, self.bands.shape[0], self.patch_size, self.patch_size)).to(next(model.parameters()).device)
        else:
            dummy = torch.zeros((self.batch_size, len(self.descriptors), self.patch_size, self.patch_size)).to(next(model.parameters()).device)
        dummy_preds = model(dummy, [self.descriptors for _ in range(dummy.shape[0])])
        return dummy_preds.shape[1]

    def predict(self, model):

        class_num = self.get_class_num(model)

        # Predict all patches
        mask = np.zeros((class_num, *self.bordered_bands.shape[1:]), dtype=np.float32)
        count = np.zeros((1, *self.bordered_bands.shape[1:]), dtype=np.float32)

        if self.verbose:
            iter = tqdm(range(0, len(self.coords), self.batch_size), desc='Predicting patches')
        else:
            iter = range(0, len(self.coords), self.batch_size)

        for i in iter:
            batch_coords = self.coords[i:i+self.batch_size]
            batch = self.get_patches(batch_coords)
            batch = np.stack(batch, axis=0)
            batch = torch.from_numpy(batch.astype('float32')).to(next(model.parameters()).device)
            if self.descriptors is None:
                preds = model(batch)
            else:
                preds = model(batch, [self.descriptors for _ in range(batch.shape[0])])

            mask = self.add_patches(mask, preds, batch_coords)
            count = self.add_patches(count, torch.ones((self.batch_size, *preds.shape[2:])), batch_coords)
        
        # Remove borders
        offsets_y, offsets_x = self.offsets
        mask = mask[:, offsets_y[0]:-offsets_y[1], offsets_x[0]:-offsets_x[1]]
        count = count[:, offsets_y[0]:-offsets_y[1], offsets_x[0]:-offsets_x[1]]

        if np.any(count == 0):
            print('Warning: some pixels were never predicted. Is your stride bigger than your patch size? Setting these pixels to nan.')

        mask /= (count+1e-6)
        # mask[count == 0] = np.nan # Set unlabelled pixels to nan


        return mask

