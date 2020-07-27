#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:30:54 2019

@author: paskali
"""

''' Train images augmentation.

    Load images and binary masks from train folder, and apply various methods 
    for transformation. Finally save them in the train folder.
'''

import os, nrrd, time
import numpy as np
import elasticdeform
from scipy import ndimage

def augment_images(train_folder='data/train'):
    '''
    Generate augmented images and binary mask. Then save them in train_folder.  
    The following ratio is used: one rotation, three transversal elastic 
    deformation, two combination of both. Finally generate median, mean, 
    Gaussian filter to all images and save them in train_folder.

    Parameters
    ----------
    train_folder : str, optional
        path to the folder with training images, it should have two subfolders:
            "image" and "mask". The default is 'data/train'.

    Returns
    -------
    None.

    '''
    image_names = os.listdir(os.path.join(train_folder, 'image'))
    
    print('Augmenting images...')
    counter = 1
    start = time.time()
    for name in image_names:
        print(f'Augmenting image...[{counter}/{len(image_names)}]')
        image_path = os.path.join(train_folder, 'image', name)
        mask_path = os.path.join(train_folder, 'mask', name)
        image = nrrd.read(image_path)[0]
        mask = nrrd.read(mask_path)[0]
        
        image_path = image_path.split('.')[0]
        mask_path = mask_path.split('.')[0]
        
        apply_rotation(image, mask, image_path, mask_path, [np.random.randint(-20,20)])
        elastic_deform(image, mask, image_path, mask_path, 3)
        rot_and_elastic_deform(image, mask, image_path, mask_path, 2)
        
        counter += 1
    apply_filters(train_folder)
    end = time.time()
    print(f'Duration: {time.strftime("%T", time.gmtime(end-start))}') 


def apply_rotation(image, mask, image_path, mask_path, angle_list):
    """
    Apply rotation for every element in angle_list to image
    and binary mask. Then save them to corresponding image_path and mask_path.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.
    angle_list : list
        list with degrees of rotation.

    Returns
    -------
    None.

    """
    counter = 0
    for i in angle_list:
        nrrd.write(f'{image_path}_ro{i}.nrrd', _fix_image_size(ndimage.rotate(image, i, cval=image.min()), image.shape))
        nrrd.write(f'{mask_path}_ro{i}.nrrd', _fix_image_size(ndimage.rotate(mask, i), mask.shape))
        
        counter += 1
        print(f'Rotating ({i})...[{counter/len(angle_list):.0%}]')


def elastic_deform(image, mask, image_path, mask_path, times, sigma=4):
    """
    Apply transversal elastic deformation to each slide of image
    and binary mask. Then save them to corresponding image_path and mask_path.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.
    times : int
        the number to repeat deformation.
    sigma : int
        used for elastic deformation. The default is 4.
        lower value for images with lower resolution, could be higher for images
        with higher resolution.
        
        E.g.:
        3-4 sigma for resolution of 128 x 128
        15-20 sigma for resolution of 512 x 512

    Returns
    -------
    None.

    """
    counter = 0
    for i in range(times):
        # Sigma 20 is okay for (512,512), but for lower resolution it should be lower.
        # Use 3 for smoother deformation, and 5 for stronger deformations.
        image, mask = elasticdeform.deform_random_grid([image, mask], sigma=sigma, points=3, cval=image.min(), prefilter=False, axis=(0,1))
        mask = np.where(mask != 1, 0, 1)
        
        nrrd.write(f'{image_path}_ed{i}.nrrd', image)
        nrrd.write(f'{mask_path}_ed{i}.nrrd', mask)
        
        counter += 1
        print(f'Elastic deformation...[{counter/times:.0%}]')

def rot_and_elastic_deform(image, mask, image_path, mask_path, times, angle_range=(-20,20), sigma=4):
    """
    Apply random rotation to image and binary mask in range from -20 to +20 
    degrees. Then apply transversal elastic deformation to each slide of image
    and binary mask. Finally save them to corresponding image_path and mask_path.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.
    times : int
        the number to repeat the functions.
    angle_range : tuple, optional
        range of min and max degrees. The default is (-20,20).
    sigma : int, optional
        used for elastic deformation. The default is 4.
        lower value for images with lower resolution, could be higher for images
        with higher resolution.
        
        E.g.:
        3-4 sigma for resolution of 128 x 128
        15-20 sigma for resolution of 512 x 512

    Returns
    -------
    None.

    """
    counter = 0
    for i in range(times):
        angle = np.random.randint(angle_range)
        # Rotation
        image = _fix_image_size(ndimage.rotate(image, angle, cval=image.min()), image.shape)
        mask = _fix_image_size(ndimage.rotate(mask, angle), mask.shape)
        
        # Elastic deformation
        image, mask = elasticdeform.deform_random_grid([image, mask], sigma=sigma, points=3, cval=image.min(), prefilter=False, axis=(0,1))
        mask = np.where(mask != 1, 0, 1)
        
        nrrd.write(f'{image_path}_roed{i}.nrrd', image)
        nrrd.write(f'{mask_path}_roed{i}.nrrd', mask)
        
        counter += 1
        print(f'Rotation/Elas.deform...[{counter/times:.0%}]')

def random_zoom(image, mask, image_path, mask_path, zoom_list=np.random.uniform(0.8, 1.2, size=3)):
    """
    Randomly resize image and binary mask by zoom factor in range from 0.8 to 1.2 
    or by predefined list with zoom factors.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.
    zoom_list : list, optional
        list with zoom factors. The default is np.random.uniform(0.8, 1.2, size=3).

    Returns
    -------
    None.

    """
    counter = 0 
    for i in zoom_list:
        nrrd.write(f'{image_path}_z{counter}.nrrd', _fix_image_size(ndimage.zoom(image, i), image.shape))
        nrrd.write(f'{mask_path}_z{counter}.nrrd', _fix_image_size(ndimage.zoom(mask, i), mask.shape))
        
        counter += 1
        print(f'Random Zooming ({i:.2f})...[{counter/len(zoom_list):.0%}]')

def random_shift(image, mask, image_path, mask_path, times):
    """
    Randomly shift image and binary mask in range X = [-10,10] and Y = [-10,10]

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.
    times : int
        the number of shift repeat.

    Returns
    -------
    None.

    """
    counter = 0
    for i in range(times):
        x_shift, y_shift, z_shift = (np.random.randint(-10,10), np.random.randint(-10,10), 0)
                                    
        nrrd.write(f'{image_path}_shift{i}.nrrd', ndimage.shift(image, (x_shift, y_shift, z_shift)))
        nrrd.write(f'{mask_path}_shift{i}.nrrd', ndimage.shift(mask, (x_shift, y_shift, z_shift)))
        
        counter += 1
        print(f'Random shifting {(x_shift, y_shift, z_shift)}...[{counter/times:.0%}]')

def filter_images(image, mask, image_path, mask_path):
    """
    Apply median, mean, Gaussian filter only to image. Then save image and 
    binary mask to corressponding image_path and mask_path.

    Parameters
    ----------
    image : nparray
        3D nparray of image.
    mask : nparray
        3D nparray of binary mask.
    image_path : str
        path to where the image should be saved.
    mask_path : str
        path to where the mask should be saved.

    Returns
    -------
    None.

    """
    nrrd.write(f'{image_path}_medf.nrrd', ndimage.median_filter(image, size=(3,3,3)))
    nrrd.write(f'{image_path}_meanf.nrrd', ndimage.uniform_filter(image, size=(3,3,3)))
    nrrd.write(f'{image_path}_gauss.nrrd', ndimage.gaussian_filter(image, sigma=1))
    
    nrrd.write(f'{mask_path}_medf.nrrd', mask)
    nrrd.write(f'{mask_path}_meanf.nrrd', mask)
    nrrd.write(f'{mask_path}_gauss.nrrd', mask)
    print('Applying Filters...[100%]')
    

def apply_filters(train_folder):
    """
    Load all images and binary masks in train_folder and generate three images with
    median, mean and Gaussian filter for each.

    Parameters
    ----------
    train_folder : str
        path to train folder containing subfolders with images and masks.

    Returns
    -------
    None.

    """
    name_list = os.listdir(os.path.join(train_folder, 'image'))
    
    counter = 0
    for name in name_list:
        counter += 1
        image_path = os.path.join(train_folder, 'image', name)
        mask_path = os.path.join(train_folder, 'mask', name)
        
        image = nrrd.read(image_path)[0]
        mask = nrrd.read(mask_path)[0]
        
        image_path, mask_path = image_path.split('.')[0], mask_path.split('.')[0]
    
        nrrd.write(f'{image_path}_medf.nrrd', ndimage.median_filter(image, size=(3,3,3)))
        nrrd.write(f'{image_path}_meanf.nrrd', ndimage.uniform_filter(image, size=(3,3,3)))
        nrrd.write(f'{image_path}_gauss.nrrd', ndimage.gaussian_filter(image, sigma=1))
        
        nrrd.write(f'{mask_path}_medf.nrrd', mask)
        nrrd.write(f'{mask_path}_meanf.nrrd', mask)
        nrrd.write(f'{mask_path}_gauss.nrrd', mask)
        
        print(f'\r[Applying filter...{counter/len(name_list):.2%}]', end='')
    print()

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

if __name__ == '__main__':
    augment_images('data/train')
