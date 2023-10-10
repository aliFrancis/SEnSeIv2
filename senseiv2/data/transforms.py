import numpy as np
import random

class Base:
    """
    Makes transformation for image/mask pair that is a randomly cropped, rotated
    and flipped portion of the original.

    Parameters
    ----------
    patch_size : int
        Spatial dimension of output image/mask pair (assumes Width==Height).
    fixed : bool, optional
        If True, always take patch from top-left of scene, with no rotation or
        flipping. This is useful for validation and reproducability.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self, patch_size, fixed = False):
        self.patch_size = patch_size
        self.fixed = fixed

    def __call__(self,img, mask, metadata):
        if self.fixed:
            left = 0
            top = 0
            crop_size = int(
                min(self.patch_size, img.shape[0], img.shape[1]))
            img = img[top:top + crop_size, left:left + crop_size, ...]
            mask = mask[top:top + crop_size, left:left + crop_size, ...]

        else:
            if not self.patch_size == img.shape[0]:
                crop_size = int(
                    min(self.patch_size, img.shape[0] - 1, img.shape[1] - 1))

                left = int(
                    random.randint(
                        0,
                        img.shape[1] -
                        crop_size))
                top = int(
                    random.randint(
                        0,
                        img.shape[0] -
                        crop_size))

                img = img[top:top + crop_size, left:left + crop_size, ...]
                mask = mask[top:top + crop_size, left:left + crop_size, ...]

        rota = random.choice([0, 1, 2, 3])
        flip = random.choice([True, False])
        if rota and not self.fixed:
            img = np.rot90(img, k=rota)
            mask = np.rot90(mask, k=rota)
        if flip and not self.fixed:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        return img, mask, metadata

class Class_map:
    """
    Converts mask with N classes into a mask with M classes.

    Parameters
    ----------
    class_map : Dict
        Dictionary mapping original classes to new classes.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,class_map):
        self.class_map = class_map
        self.new_classes = list(set(class_map.keys()))

    def __call__(self,img, mask, metadata):
        new_mask = np.zeros(mask.shape[:-1]+(len(self.new_classes),))
        for i in range(len(self.new_classes)):            
            new_mask_i = np.sum(mask[...,self.class_map[self.new_classes[i]]],axis=-1) > 0
            new_mask[...,i] = new_mask_i

        metadata['classes'] = self.new_classes
        return img, new_mask, metadata


class Sometimes:
    """
    Wrapper function which randomly applies the transform with probability p.

    Parameters
    ----------
    p : float
        Probability of transform being applied
    transform : func
        Function which transforms image/mask/metadata triplets.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """

    def __init__(self,p, transform):
        self.p = p
        self.transform = transform

    def __call__(self,img, mask, metadata):
        random_apply = random.random() < self.p
        if random_apply:
            return self.transform(img, mask, metadata)
        else:
            return img, mask, metadata

class Chromatic_shift:
    """
    Adds a different random amount to each spectral band in image.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,shift_min=-0.10, shift_max=0.10):
        self.shift_min=shift_min
        self.shift_max=shift_max

    def __call__(self,img, mask, metadata):
        img = img + np.random.uniform(
                    low=self.shift_min,
                    high=self.shift_max, 
                    size=[1, 1, img.shape[-1]]
                    )
        return img, mask, metadata

class Chromatic_scale:
    """
    Multiplies each spectral band by a different random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,factor_min=0.90, factor_max=1.10):
        self.factor_min=factor_min
        self.factor_max=factor_max

    def __call__(self,img, mask, metadata):
        img = img * np.random.uniform(low=self.factor_min,
                                      high=self.factor_max, size=[1, 1, img.shape[-1]])
        return img, mask, metadata

class Intensity_shift:
    """
    Adds single random amount to all spectral bands.

    Parameters
    ----------
    shift_min : float, optional
        Lower bound for random shift.
    shift_max : float, optional
        Upper bound for random shift.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,shift_min=-0.10, shift_max=0.10):
        self.shift_min=shift_min
        self.shift_max=shift_max
    def __call__(self,img, mask, metadata):
        img = img + (self.shift_max-self.shift_min)*random.random()+self.shift_min
        return img, mask, metadata

class Intensity_scale:
    """
    Multiplies all spectral bands by a single random factor.

    Parameters
    ----------
    factor_min : float, optional
        Lower bound for random factor.
    factor_max : float, optional
        Upper bound for random factor.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """

    def __init__(self,factor_min=0.95, factor_max=1.05):
        self.factor_min=factor_min
        self.factor_max=factor_max
    def __call__(self,img, mask, metadata):
        img = img * random.uniform(self.factor_min, self.factor_max)
        return img, mask, metadata

class White_noise:
    """
    Adds white noise to image.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of white noise

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,sigma=0.1):
        self.sigma=sigma
    def __call__(self,img, mask, metadata):
        noise = (np.random.randn(*img.shape) * self.sigma)
        return img + noise, mask, metadata

class Bandwise_salt_and_pepper:
    """
    Adds salt and pepper (light and dark) noise to image,  treating each band independently.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,salt_rate, pepp_rate, pepp_value=0, salt_value=255):
        self.salt_rate  = salt_rate
        self.pepp_rate  = pepp_rate
        self.pepp_value = pepp_value
        self.salt_value = salt_value
    def __call__(self,img, mask, metadata):
        salt_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - self.salt_rate, self.salt_rate])
        pepp_mask = np.random.choice([False, True], size=img.shape, p=[
                                     1 - self.pepp_rate, self.pepp_rate])

        # img[salt_mask] = self.salt_value
        # img[pepp_mask] = self.pepp_value

        return img, mask, metadata

class Salt_and_pepper:
    """
    Adds salt and pepper (light and dark) noise to image, to all bands in a pixel.

    Parameters
    ----------
    salt_rate : float
        Percentage of pixels that are set to salt_value.
    pepp_rate : float
        Percentage of pixels that are set to pepp_value.
    pepp_value : float, optional
        Value that pepper pixels are set to.
    salt_value : float, optional
        Value that salt pixels are set to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,salt_rate, pepp_rate, pepp_value=0, salt_value=255):
        self.salt_rate  = salt_rate
        self.pepp_rate  = pepp_rate
        self.pepp_value = pepp_value
        self.salt_value = salt_value
    def __call__(self,img, mask, metadata):
        salt_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - self.salt_rate, self.salt_rate])
        pepp_mask = np.random.choice(
            [False, True], size=img.shape[:-1], p=[1 - self.pepp_rate, self.pepp_rate])

        # img[salt_mask] = [self.salt_value for i in range(img.shape[-1])]
        # img[pepp_mask] = [self.pepp_value for i in range(img.shape[-1])]

        return img, mask, metadata

class Quantize:
    """
    Quantizes an image based on a given number of steps by rounding values to closest
    value.

    Parameters
    ----------
    step_size : float
        Size of step to round values to.

    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """
    def __init__(self,step_size):
        self.stepsize = step_size

    def __call__(self,img, mask, metadata):
        img = (img//self.stepsize)*self.stepsize
        return img, mask, metadata

class No_data_edges:
    """
    Adds a random number of pixels of no data to the edges of the image.

    Parameters
    ----------
    N : int, optional
        Max number of pixels to add to each edge.
    p : float, optional
        Probability of adding pixels to each edge.
    
    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """

    def __init__(self,N=20,p=0.5):
        self.N = N
        self.p = p

    def __call__(self, img, mask, metadata):

        # Find fill class position in metadata
        fill_class = metadata['classes'].index('no_data')
        fill_class_vector = np.zeros(mask.shape[-1])
        fill_class_vector[fill_class] = 1

        # Randomly choose number of pixels to fill on each edge
        N = np.random.randint(0,self.N+1)

        # Randomly choose whether to fill pixels to each edge
        add_top = np.random.choice([True,False],p=[self.p,1-self.p])
        add_bottom = np.random.choice([True,False],p=[self.p,1-self.p])
        add_left = np.random.choice([True,False],p=[self.p,1-self.p])
        add_right = np.random.choice([True,False],p=[self.p,1-self.p])

        # Fill pixels on each edge, using copy as to not modify original image
        img_copy = img.copy()
        mask_copy = mask.copy()

        if add_top:
            img_copy[:N] = 0
            mask_copy[:N,:] = fill_class_vector
        if add_bottom:
            img_copy[-N:] = 0
            mask_copy[-N:,:] = fill_class_vector
        if add_left:
            img_copy[:,:N] = 0
            mask_copy[:,:N] = fill_class_vector
        if add_right:
            img_copy[:,-N:] = 0
            mask_copy[:,-N:] = fill_class_vector

        return img_copy, mask_copy, metadata

class No_data_rectangles:
    """
    Adds a random set of rectangles of no data to the image.

    Parameters
    ----------
    N : int, optional
        Max number of pixels to add to each edge.
    p : float, optional
        Probability of adding pixels to each edge.
    
    Returns
    -------
    apply_transform : func
        Transformation for image/mask/metadata triplets.
    """

    def __init__(self,N=20,scale=0.1,p=0.5):
        self.N = N
        self.p = p

    def __call__(self, img, mask, metadata):

        # Find fill class position in metadata
        # try:
        #     fill_class = metadata['classes'].index('fill')
        # except:
        #     raise Exception('No fill class in metadata')
        fill_class = -1 # HACK, LAZY, FIX
        fill_class_vector = np.zeros(mask.shape[-1])
        fill_class_vector[fill_class] = 1        

        # Randomly choose whether to fill pixels to each edge
        add_top = np.random.choice([True,False],p=[self.p,1-self.p])
        add_bottom = np.random.choice([True,False],p=[self.p,1-self.p])
        add_left = np.random.choice([True,False],p=[self.p,1-self.p])
        add_right = np.random.choice([True,False],p=[self.p,1-self.p])

        # Fill pixels on each edge, using copy as to not modify original image
        img_copy = img.copy()
        mask_copy = mask.copy()

        if add_top:
            N = np.random.randint(0,self.N+1)
            img_copy[:N] = 0
            mask_copy[:N,:] = fill_class_vector
        if add_bottom:
            N = np.random.randint(0,self.N+1)
            img_copy[-N:] = 0
            mask_copy[-N:,:] = fill_class_vector
        if add_left:
            N = np.random.randint(0,self.N+1)
            img_copy[:,:N] = 0
            mask_copy[:,:N] = fill_class_vector
        if add_right:
            N = np.random.randint(0,self.N+1)
            img_copy[:,-N:] = 0
            mask_copy[:,-N:] = fill_class_vector

        return img_copy, mask_copy, metadata