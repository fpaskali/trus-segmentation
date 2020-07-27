#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:42:17 2019

@author: paskali
"""
import os, nrrd, shutil, json
import numpy as np
from scipy import ndimage

class ImageProcessor(object):
    
    def __init__(self, input_folder='data/raw_train', test_set=False):
        """
        Regenerate root folder structure in data folder, that is used later by other
        scripts. Then load images from input_folder. 
        
        Root folder structure:
            data --> train ----> image
                     |           mask
                     val ------> image
                     |           mask
                     test -----> image
                     |           mask
                     results
                     
        Caution!: It removes everything included in folders. Results should be
        saved, before initialization of the ImageProcessor.

        Parameters
        ----------
        input_folder : str, optional
            Folder containing unprocessed images. The default is 'data/raw_train'.
        test_set : bool, optional
            If True, load only images without binary masks. The default is False.

        Returns
        -------
        None.

        """
        
        self.input_folder = input_folder
        self.test_set = test_set
        self.image_names = []
        self.images = []
        self.masks = []
        
        self.create_root_skeleton()
        print("Loading images...")
        self._read_image_names()
        self._print_image_names()
        self._load_images()
        
    def _read_image_names(self):
        """
        Read NRRD image names from input_folder attribute and store them in 
        image_names attribute.

        Returns
        -------
        None.

        """
        names = []
        for file in os.listdir(os.path.join(self.input_folder, 'image')):
            names.append(file)
        names.sort()
        self.image_names = names
            
    def _print_image_names(self):
        """
        Print all image names in image_names attribute.

        Returns
        -------
        None.

        """
        for name in self.image_names:
            print(name)
    
    def _load_images(self):
        """
        Load NRRD images and binary masks as numpy array.

        Returns
        -------
        None.

        """
        if not self.test_set:
            images = []
            masks = []
            for item in self.image_names:
                image = nrrd.read(os.path.join(self.input_folder, 'image', item))[0]
                mask = nrrd.read(os.path.join(self.input_folder, 'mask', item))[0]
                
                images.append(image)
                masks.append(mask)
                
            self.images = images
            self.masks = masks
        else:
            images = []
            for item in self.image_names:
                image = nrrd.read(os.path.join(self.input_folder, 'image', item))[0]           
                images.append(image)
              
            self.images = images            
        
        print(f"Loaded {len(self.images)} images and {len(self.masks)} binary masks.")
    
    def normalize(self, norm_params=None):
        """
        Z-score normalization to mean = 0, std = 1.

        Parameters
        ----------
        norm_params : json, optional
            If loaded use mean and std stored in json file, if not generate 
            norm_params.json with "mean" = mean of all images, "std" = std of all images.

        Returns
        -------
        None.

        """
        images = np.asarray(self.images)
        images = images.astype(np.float32)
        
        if not norm_params:
            std = np.std(images)
            mean = np.mean(images)
            if std > 0:
                self.images = (images - mean) / std
            else:
                self.images = images * 0.
                
            with open('norm_params.json', 'w') as file:
                json.dump({'mean':float(mean), 'std':float(std)}, file)
        
        else:
            with open(norm_params, 'r') as file:
                norm_params = json.load(file)
                std = norm_params['std']
                mean = norm_params['mean']
            if std > 0:
                self.images = (images - mean) / std
            else:
                self.images = images * 0.
            
               
    def resize_images(self, target_size):
        """
        Resize images and binary mask to target size.

        Parameters
        ----------
        target_size : tuple
            E.g. target_size = (128,128,128).

        Returns
        -------
        None.

        """
        
        if not self.test_set:
            images = []
            masks = []
            for image, mask in zip(self.images, self.masks):
                zoom_size = target_size[0]/image.shape[0], target_size[1]/image.shape[0], target_size[2]/image.shape[2]              
                images.append(self._crop_image(ndimage.zoom(image, zoom_size),target_size))
                masks.append(self._crop_image(ndimage.zoom(mask, zoom_size),target_size))
            
            self.images = images
            self.masks = masks
        else:
            images = []
            for image in self.images:
                images.append(self._crop_image(ndimage.zoom(image, zoom_size),target_size))
            
            self.images = images
    
    def crop_images(self, target_size):
        """
        Crop the region of interest with target_size. The region is always 
        centered in original image.

        Parameters
        ----------
        target_size : tuple
            E.g. target_size = (128,128,128).

        Returns
        -------
        None.

        """
        if not self.test_set:
            images = []
            masks = []
            for image, mask in zip(self.images, self.masks):
                images.append(self._crop_image(image, target_size))
                masks.append(self._crop_image(mask, target_size))
            
            self.images = images
            self.masks = masks

        else:
            images = []
            for image in self.images:
                images.append(self._crop_image(image, target_size))
            
            self.images = images
        
        
    def _crop_image(self, image, target_size):
        ''' Crop original image to target size. If any axis size is smaller than
            original, add padding to the original image to make it equal to 
            target axis size.
        '''
        org_x, org_y, org_z = image.shape
        target_x, target_y, target_z = target_size
        
        if target_x > org_x:
            modulo = (target_x - org_x) % 2
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
        
    def create_root_skeleton(self, preserve_results=False):
        """
        Generate new empty root folder structure.

        Parameters
        ----------
        preserve_results : bool, optional
            If preserve_results is True, it do not regenerate 'data/results'. 
            The default is False.

        Returns
        -------
        None.

        """
        
        root_folders = ['data/train', 'data/val', 'data/test', 'data/results']
        folder_tree = ['data/train',
               'data/train/image',
               'data/train/mask',
               'data/val',
               'data/val/image',
               'data/val/mask',
               'data/test',
               'data/test/image',
               'data/test/mask',
               'data/results']
        
        if preserve_results:
            root_folders.remove('data/results')
            folder_tree.remove('data/results')
            
        for i in root_folders:
            if os.path.exists(i):
                shutil.rmtree(i)

        for i in folder_tree:
            os.mkdir(i)
    
    def save_images(self, output_folder, index_list=None):
        """
        Save images and binary masks in output_folder.

        Parameters
        ----------
        output_folder : str
            path to directory where images should be saved.
        index_list : list, optional
            if index_list = None, it saves all images and binary mask to output_folder.
            if index_list is defined, it saves only images and mask with defined
            indices in the list. The default is None.
        test : bool, optional
            If True, resizes only the image, without binary mask. The default is False.

        Returns
        -------
        None.

        """       
        if not self.test_set:
            if not os.path.exists(output_folder):
                os.makedirs(os.path.join(output_folder,'image'))
                os.makedirs(os.path.join(output_folder,'mask'))
            if not index_list:
                for i in range(len(self.image_names)):
                    nrrd.write(os.path.join(output_folder, 'image', self.image_names[i]), self.images[i])
                    nrrd.write(os.path.join(output_folder, 'mask', self.image_names[i]), self.masks[i])
            else:
                for i in index_list:
                    nrrd.write(os.path.join(output_folder, 'image', self.image_names[i]), self.images[i])
                    nrrd.write(os.path.join(output_folder, 'mask', self.image_names[i]), self.masks[i])
        else:
            if not os.path.exists(output_folder):
                os.makedirs(os.path.join(output_folder,'image'))
            if not index_list:
                for i in range(len(self.image_names)):
                    nrrd.write(os.path.join(output_folder, 'image', self.image_names[i]), self.images[i])
            else:
                for i in index_list:
                    nrrd.write(os.path.join(output_folder, 'image', self.image_names[i]), self.images[i])

