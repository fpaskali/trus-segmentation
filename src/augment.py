#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:30:54 2019

@author: paskali
"""

""" 
Train images augmentation.

Load images and binary masks from train folder, and apply various methods 
for transformation. Finally save them in the train folder.
"""

import random, time, csv, os
import numpy as np
import elasticdeform
from scipy import ndimage
import tensorflow as tf

def rotation(image, mask):
    """
    Apply rotation to image and binary mask.

    Parameters
    ----------
    image : numpy array
        3D numpy array of image.
    mask : numpy array
        3D numpy array of binary mask.

    Returns
    -------
    numpy array, numpy array
        rotated image and binary mask.

    """
    angle = np.random.randint(-20,20)
    
    return _fix_image_size(ndimage.rotate(image, angle, cval=image.min()), image.shape), _fix_image_size(ndimage.rotate(mask, angle), mask.shape)

def elastic_deform(image, mask, sigma=4):
    """
    Apply transversal elastic deformation to each slide of image
    and binary mask. Then save them to corresponding image_path and mask_path.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    sigma : int
        used for elastic deformation. The default is 4.
        lower value for images with lower resolution, could be higher for images
        with higher resolution.
        
        E.g.:
        3-4 sigma for resolution of 128 x 128
        15-20 sigma for resolution of 512 x 512

    Returns
    -------
    numpy array, numpy array
        deformed image and binary mask.
    """
    # Sigma 20 is okay for (512,512), but for lower resolution it should be lower.
    # Use 3 for smoother deformation, and 5 for stronger.
    image, mask = elasticdeform.deform_random_grid([image, mask], sigma=sigma, points=3, cval=image.min(), prefilter=False, axis=(0,1))
    mask = np.where(mask != 1, 0, 1)
    
    return image, mask

def random_zoom(image, mask):
    """
    Randomly resize image and binary mask by zoom factor in range from 0.8 to 1.2

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.

    Returns
    -------
    numpy array, numpy array
        resized image and binary mask.

    """
    zoom = np.random.uniform(0.8, 1.2)
    
    return _fix_image_size(ndimage.zoom(image, zoom), image.shape), _fix_image_size(ndimage.zoom(mask, zoom), mask.shape)


def random_shift(image, mask):
    """
    Randomly shift image and binary mask in range X = [-10,10] and Y = [-10,10]

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.

    Returns
    -------
    numpy array, numpy array
        shifted image and binary mask.

    """
    x_shift, y_shift, z_shift = (np.random.randint(-10,10), np.random.randint(-10,10), 0)
                   
    return _fix_image_size(ndimage.shift(image, (x_shift, y_shift, z_shift))), _fix_image_size(ndimage.shift(mask, (x_shift, y_shift, z_shift)))


def mean_filter(image, mask):
    '''
    Apply mean filter.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.

    Returns
    -------
    numpy array, numpy array
        shifted image and binary mask.
    '''
    return ndimage.uniform_filter(image, size=(3,3,3)), mask

def median_filter(image, mask):
    '''
    Apply median filter.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.

    Returns
    -------
    numpy array, numpy array
        shifted image and binary mask.
    '''
    return ndimage.median_filter(image, size=(3,3,3)), mask

def gauss_filter(image, mask):
    '''
    Apply gaussian filter.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.

    Returns
    -------
    numpy array, numpy array
        shifted image and binary mask.
    '''
    return ndimage.gaussian_filter(image, sigma=1), mask

def _fix_image_size(image, target_size):
    """
    Crop 3D image to target size. If any axis size is lower than
    target size, add padding to reach target size.

    Parameters
    ----------
    image : nparray
        3D nparray.
    target_size : tuple
        tuple with value for every axis.

    Returns
    -------
    nparray
        cropped image with target size.

    """
    org_x, org_y, org_z = image.shape
    target_x, target_y, target_z = target_size
    
    if target_x > org_x:
        modulo  = (target_x - org_x) % 2
        offset = (target_x - org_x) // 2
        image = np.pad(image, ((offset, offset + modulo),(0,0),(0,0)), mode='constant')
            
    if target_y > org_y:
        modulo  = (target_y - org_y) % 2
        offset = (target_y - org_y) // 2
        image = np.pad(image, ((0,0),(offset, offset + modulo),(0,0)), mode='constant')
            
    if target_z > org_z:
        modulo  = (target_z - org_z) % 2
        offset = (target_z - org_z) // 2
        image = np.pad(image, ((0,0),(0,0),(offset, offset + modulo)), mode='constant')
            
    org_x, org_y, org_z = image.shape
    off_x, off_y, off_z = (org_x - target_x)//2, (org_y - target_y)//2, (org_z - target_z)//2
    minx, maxx = off_x, target_x + off_x
    miny, maxy = off_y, target_y + off_y
    minz, maxz = off_z, target_z + off_z

    return image[minx:maxx, miny:maxy, minz:maxz]

def augment_generator_probability(train_ds, factor, rotate_p, deform_p, filters_p, epochs,
                                  mean_filter_p=0.33, median_filter_p=0.33, gauss_filter_p=0.33):
    """
    Generator that yields augmented images. The augmentation is performed according to
    probability values, increasing the dataset by defined factor.
    
    Saves a report of augmentation in /logs directory.

    Parameters
    ----------
    train_ds : tuple
        tuple containing image and binary mask.
    factor : int
        the factor by which the sample will be increased (E.g. final sample size = factor * train sample size).
    rotate_p : float
        the probability of rotation.
    deform_p : float
        the probability of deformation.
    filters_p : float
        the probability to apply filters.
    epochs : int
        the number of sets of images to be generated.
    mean_filter_p : TYPE, optional
        The probability to apply mean filter. The default is 0.33.
    median_filter_p : TYPE, optional
        The probability to apply median filter. The default is 0.33.
    gauss_filter_p : TYPE, optional
        The probability to apply gaussian filter. The default is 0.33.

    Yields
    ------
    image : tensor
        tensor of the image.
    mask : tensor
        tensor of the mask.

    """
    if not os.path.exists("./logs"):
        os.mkdir("logs")
    log_name = f'logs/aug_{time.strftime("%Y%m%d%H%M",time.localtime())}.log'
    with open(log_name, 'w', newline='') as csvfile:
        fieldnames = ['rotate', 'deform', 'filters', 'filter']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
    for _ in range(epochs):
        for x, y in train_ds:
            for i in range(factor):
                inst={'rotate':'off',
                      'deform':'off',
                      'filters':'off',
                      'filter':'no'}
                image, mask = x, y
                if random.random() < rotate_p:
                    image, mask = rotation(image, mask)
                    inst['rotate'] = 'on'
                if random.random() < deform_p:
                    image, mask = elastic_deform(image, mask)
                    inst['deform'] = 'on'
                if random.random() < filters_p:
                    inst['filters'] = 'on'
                    chance = random.random()
                    if chance < mean_filter_p:
                        image, mask = mean_filter(image, mask)
                        inst['filter'] = 'mean'
                    elif chance < mean_filter_p + median_filter_p:
                        inst['filter'] = 'median'
                        image, mask = median_filter(image, mask)
                    else:
                        inst['filter'] = 'gauss'
                        image, mask = gauss_filter(image, mask)
                
                with open(log_name, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=list(inst.keys()))
                    writer.writerow(inst)
                
                image = np.reshape(image, image.shape + (1,))
                mask = np.reshape(mask, mask.shape + (1,))
                image = np.reshape(image, (1,) + image.shape)
                mask = np.reshape(mask, (1,) + mask.shape)
                image = tf.convert_to_tensor(image)
                mask = tf.convert_to_tensor(mask)
                    
                yield image, mask
