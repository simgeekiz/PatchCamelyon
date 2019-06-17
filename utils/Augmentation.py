import numpy as np
import cv2
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageFilter, ImageEnhance 
import random

mean = 0
var = 10
sigma = 0.1

def mirroring(image):
    a = random.randint(0,2)
    if a == 0:
        im_aug = np.fliplr(np.flipud(image))
    elif a == 1:
        im_aug = np.flipud(image)
    elif a == 2:
        im_aug = np.fliplr(image)
    return im_aug

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==3

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape_2d = image[:,:,0].shape

    dx = gaussian_filter((random_state.rand(*shape_2d) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape_2d) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape_2d[0]), np.arange(shape_2d[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    r = map_coordinates(image[:,:,0], indices, order=1).reshape(shape_2d)
    g = map_coordinates(image[:,:,1], indices, order=1).reshape(shape_2d)
    b = map_coordinates(image[:,:,2], indices, order=1).reshape(shape_2d)
   
    return np.stack((r,g,b), axis=2)

def gaussian_noise(image, sigma):
    gaussian = np.random.normal(mean, sigma, image[:,:,0].shape)
    noisy_image = np.zeros(image.shape, np.float32)

    if len(image.shape) == 2:
        noisy_image = image + gaussian
    else:
        noisy_image[:, :, 0] = image[:, :, 0] + gaussian
        noisy_image[:, :, 1] = image[:, :, 1] + gaussian
        noisy_image[:, :, 2] = image[:, :, 2] + gaussian

    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    
    return noisy_image

def Augmentation(image, rot=[0, 90], elast_a=[80, 120], elast_sig=[9.0 , 11.0], g_noise=[0,0.1],
                g_blurr=[0,0.9], brigh=[0.65,1.35], contr=[0.5,1.5], HSV=0.1, HED=0.02):
    
    st = random.randint(0,2)
       
    if st == 0:
        #### Group 1: Basic -> Rotation or Mirroring ###
        t = random.randint(0,1)
        if t == 0:
            # random rotation
            im = Image.fromarray(image)
            angle = random.sample(rot, 1)        
            im = im.rotate(angle[0])
            im = np.array(im)
        elif t == 1:
            # random mirroring
            im = mirroring(image)
    
    elif st == 1:
        im = image
        #### Group 2: Morphologic -> Elestic deformation,  gaussian noise or gaussian blurring ####
        t = random.randint(0,2)
        if t == 0:
            # random elastic trasform
            a = np.random.uniform(low=elast_a[0], high=elast_a[1], size=1)[0]
            s = np.random.uniform(low=elast_sig[0], high=elast_sig[1], size=1)[0]
            im = elastic_transform(im, a, s)
        elif t == 1:
            # random gaussian noise
            sigma = np.random.uniform(low=g_noise[0], high=g_noise[1], size=1)[0]
            im = gaussian_noise(im, sigma)
        elif t == 2:
            # random gaussian blurring
            sigma = np.random.uniform(low=g_blurr[0], high=g_blurr[1], size=1)[0]
            im = Image.fromarray(im)
            im = im.filter(ImageFilter.GaussianBlur(sigma))
            im = np.array(im)
        
    elif st == 2:
        im = image
        #### Group 3: Brightness or Contrast ####
        t = random.randint(0,1)
        if t == 0:
            # random brightness perturbation
            br = np.random.uniform(low=brigh[0], high=brigh[1], size=1)[0]
            im = Image.fromarray(im)
            im = ImageEnhance.Brightness(im)
            im = im.enhance(br)
            im = np.array(im)
        elif t == 1:
            # random contrast perturbation
            cr = np.random.uniform(low=contr[0], high=contr[1], size=1)[0]
            im = Image.fromarray(im)
            im = ImageEnhance.Contrast(im)
            im = im.enhance(cr)
            im = np.array(im)
    
    return im