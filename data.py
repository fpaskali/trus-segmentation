#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:37:04 2019

@author: paskali
"""
import os, nrrd, json
import numpy as np
import tensorflow as tf


class DataManager():

    def __init__(self, train_folder='data/train', val_folder='data/val', 
                 test_folder='data/test', result_folder='data/result', 
                 image_shape=None, patch_size=None, stride_size=None):
        """
        Data manager prepares the data for loading in training and testing
        function. It can load whole image volume, and load extracted patches.

        Parameters
        ----------
        train_folder : str, optional
            path to train folder. The default is 'data/train'.
        val_folder : str, optional
            path to validation folder. The default is 'data/val'.
        test_folder : str, optional
            path to test folder. The default is 'data/test'.
        result_folder : str
            DESCRIPTION. The default is 'data/result'
        image_shape : tuple
            The size of input image.
        patch_size : tuple, optional
            The size of each patch. The default is None.
        stride_size : tuple, optional
            The size of the stride. The default is None.
            
        If patch_size and stride_size are define, data manager is ready for
        patch extraction. Otherwise, its ready for whole image loading.

        Returns
        -------
        None.

        """
        
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.test_folder = test_folder
        self.result_folder = result_folder
        self.train_list = self._read_image_paths(train_folder)
        self.val_list = self._read_image_paths(val_folder)
        self.test_list = self._read_image_paths(test_folder)
        self.image_shape = image_shape
        self.patch_size = patch_size
        self.stride_size = stride_size
        
        if (patch_size and stride_size):
            self.train_size, self.val_size, self.test_size = self._calculate_dataset_size()
            self.input_size = patch_size + (1,)
        else:
            self.train_size, self.val_size, self.test_size = len(self.train_list), len(self.val_list), len(self.test_list)
            self.input_size = image_shape + (1,)
        
   
    def train_generator(self, epochs, batch_size=0):
        """
        Train generator with batch image preloading, to improve the speed
        of image reading by network. batch_size=0 reads the whole dataset.

        Parameters
        ----------
        epochs : int
            number of training epochs.
        batch_size : int, optional
            The number of images that should be preloaded in memory. The default is 0.
            If batch_size=0, whole dataset is loaded.

        Yields
        ------
        image : tensor
            image tensor.
        mask : tensor
            binary mask tensor.

        """
        if batch_size == 0: batch_size = len(self.train_list)
        for time in range(epochs):
            i = 0
            while batch_size * i <= len(self.train_list):
                tensors = []
                for item in self.train_list[batch_size*i:batch_size*i+batch_size]:
                    image = nrrd.read(os.path.join(self.train_folder, 'image', item))[0]
                    image = np.reshape(image, image.shape + (1,))
                    image = np.reshape(image, (1,) + image.shape)
                    image = tf.convert_to_tensor(image)
                    
                    mask = nrrd.read(os.path.join(self.train_folder, 'mask', item))[0]
                    mask = np.reshape(mask, mask.shape + (1,))
                    mask = np.reshape(mask, (1,) + mask.shape)
                    mask = tf.convert_to_tensor(mask)
                    
                    tensors.append((image, mask))
                
                for image, mask in tensors:
                    yield image, mask
                
                i+=1
    
    def train_patches_generator(self, epochs):
        """
        Train generator that loads all images in memory, extract patches.Finally
        yields image_patch and mask_patch.

        Parameters
        ----------
        epochs : int
            number of training epochs.

        Yields
        ------
        image_patch : tensor
            image patch tensor.
        mask_patch : tensor
            binary mask patch tensor.
        """
        # Loading images
        images = []
        masks = []
        for item in self.train_list:
            image = nrrd.read(os.path.join(self.train_folder, 'image', item))[0]               
            mask = nrrd.read(os.path.join(self.train_folder, 'mask', item))[0]

            images.append(image)
            masks.append(mask)
                
        # Patches extraction
        for _ in range(epochs):
            for i in range(len(images)):
                print('\nImage', f'[{i+1}/{len(images)}] - Epoch {_+1}/{epochs}')
                image = images[i]
                mask = masks[i]
                print('Shape', image.shape, mask.shape)
                for image_patch, mask_patch in self._extract_patches(image, mask, self.patch_size, self.stride_size):
                    image_patch = tf.convert_to_tensor(image_patch)
                    mask_patch = tf.convert_to_tensor(mask_patch)                 
                    yield image_patch, mask_patch
                    

    def _extract_patches(self, image, mask, patch_size, stride):
        """
        Patch generator.

        Parameters
        ----------
        image : nparray
            3D nparray of image.
        mask : nparray
            3D nparray of binary mask.
        patch_size : tuple
            The size of each patch.
        stride : tuple
            The size of the stride.

        Yields
        ------
        image_patch : nparray
            a patch of original image.
        mask_patch : nparray
            a matching patch of binary mask.

        """
        image_h, image_w, image_d = image.shape
        
        for z in range(0, image_d-patch_size[2]+1, stride[2]):
            for y in range(0, image_h-patch_size[1]+1, stride[1]):
                for x in range(0, image_w-patch_size[0]+1, stride[0]):
                    image_patch = np.zeros(patch_size)
                    image_slice = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    image_patch[0:image_slice.shape[0], 0:image_slice.shape[1], 0:image_slice.shape[2]] += image_slice
                    
                    mask_patch = np.zeros(patch_size)
                    mask_slice = mask[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    mask_patch[0:image_slice.shape[0], 0:image_slice.shape[1], 0:image_slice.shape[2]] += mask_slice
                    
                    image_patch = np.reshape(image_patch, image_patch.shape + (1,))
                    image_patch = np.reshape(image_patch, (1,) + image_patch.shape)
                    
                    mask_patch = np.reshape(mask_patch, mask_patch.shape + (1,))
                    mask_patch = np.reshape(mask_patch, (1,) + mask_patch.shape)
                    
                    yield image_patch, mask_patch
   
    def _count_patches(self):
        """
        Count the number of patches extracted from image with specific shape.

        Returns
        -------
        count : int
            the number of patches extracted of each image.
        """
        patch_size = self.patch_size
        stride = self.stride_size
        image_h, image_w, image_d = self.image_shape
        count = 0
        for z in range(0, image_d-patch_size[2]+1, stride[2]):
            for y in range(0, image_h-patch_size[1]+1, stride[1]):
                for x in range(0, image_w-patch_size[0]+1, stride[0]):
                    count += 1
        return count
    
    def _calculate_dataset_size(self):
        """
        Compute the total number of patches for train, validation and test set.

        Returns
        -------
        int
            Total number of pathces for train set.
        int
            Total number of pathces for validation set.
        int
            Total number of pathces for test set.
        """
        patches_nu = self._count_patches()
        return len(self.train_list) * patches_nu, len(self.val_list) * patches_nu, len(self.test_list) * patches_nu
                        
    def _extract_test_patches(self, image, image_title, patch_size, stride):
        image_h, image_w, image_d = image.shape
        
        patches = []
        patches_info = {}
        
        idx = 0
        for z in range(0, image_d-patch_size[2]+1, stride[2]):
            for y in range(0, image_h-patch_size[1]+1, stride[1]):
                for x in range(0, image_w-patch_size[0]+1, stride[0]):
                    image_patch = np.zeros(patch_size)
                    image_slice = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
                    image_patch[0:image_slice.shape[0], 0:image_slice.shape[1], 0:image_slice.shape[2]] += image_slice
                    
                    patches_info[idx] = x, y, z
                    patches.append(image_patch)
                    idx += 1
                    
        patches_info['image_res'] = image.shape
        patches_info['size'] = patch_size
        patches_info['stride'] = stride
        patches_info['len'] = idx
        
        with open(f'{self.result_folder}/{image_title}.json', 'w') as file:
            json.dump(patches_info, file)
        
        return patches

                
    def val_generator(self, epochs):
        """
        Validation set generator. Load image and binary mask, then yield
        image binary and mask tensor.

        Parameters
        ----------
        epochs : int
            number of training epochs.

        Yields
        ------
        image : tensor
            image tensor.
        mask : tensor
            mask tensor.

        """
        for i in range(epochs):
            for item in self.val_list:
                image = nrrd.read(os.path.join(self.val_folder,'image', item))[0]
                image = np.reshape(image, image.shape + (1,))
                image = np.reshape(image, (1,) + image.shape)
                image = tf.convert_to_tensor(image)
                
                mask = nrrd.read(os.path.join(self.val_folder, 'mask', item))[0]
                mask = np.reshape(mask, mask.shape + (1,))
                mask = np.reshape(mask, (1,) + mask.shape)
                mask = tf.convert_to_tensor(mask)
                
                yield (image, mask)
                
    def val_patches_generator(self, epochs):
        '''
        Validation set generator that extract and yield patches from each validation image.

        Parameters
        ----------
        epochs : int
            number of training epochs.

        Yields
        ------
        image_patch : tensor
            image patch tensor.
        mask_patch : tensor
            binary mask patch tensor.

        '''
        
        # Loading images
        images = []
        masks = []
        for item in self.val_list:
            image = nrrd.read(os.path.join(self.val_folder, 'image', item))[0]               
            mask = nrrd.read(os.path.join(self.val_folder, 'mask', item))[0]
            images.append(image)
            masks.append(mask)
                
        # Patches extraction
        for _ in range(epochs):
            for i in range(len(images)):
                for image, mask in self._extract_patches(images[i], masks[i], self.patch_size, self.stride_size):
                    image = tf.convert_to_tensor(image)
                    mask = tf.convert_to_tensor(mask)
                    yield image, mask
                    
                
    def test_generator(self):
        """
        Test set generator. Load only image, then yield
        image tensor.

        Yields
        ------
        image : tensor
            image tensor.

        """
        for item in self.test_list:
            image = nrrd.read(os.path.join(self.test_folder,'image', item))[0]
            image = np.reshape(image, image.shape + (1,))
            image = np.reshape(image, (1,) + image.shape)
            image = tf.convert_to_tensor(image, dtype=tf.float32)
            
            yield image
            
    def test_patches_generator(self):
        """
        Test set generator that extract and yield patches from each image.

        Yields
        ------
        patch : tensor
            patch tensor.

        """
        
        # Loading images
        image_names = []
        images = []
        for item in self.test_list:
            image = nrrd.read(os.path.join(self.test_folder, 'image', item))[0]               
            image_names.append(item)
            images.append(image)
        
        # Patches extraction
        for i in range(len(images)):
            for patch in self._extract_test_patches(images[i], image_names[i], self.patch_size, self.stride_size):
                patch = np.expand_dims(patch, axis=-1)
                patch = np.expand_dims(patch, axis=0)
                patch = tf.convert_to_tensor(patch, dtype=tf.float32)
                yield patch

    def _read_image_paths(self, folder):
        """
        Read the names of images in [folder]/image/. 
        Return a list of sorted names.

        Parameters
        ----------
        folder : str
            path to folder where image folder is located.

        Returns
        -------
        image_path_list : list
            list with image names.

        """
        image_path_list = os.listdir(os.path.join(folder, 'image'))
        image_path_list.sort()

        return image_path_list

    def _fuse_patches(self, patches, patches_info_json):
        '''
        Load list of patches as numpy arrays, and info about the target images as json file.
        
        Fuse the patches into target image.

        Parameters
        ----------
        patches : nparray
            nparray containing all patches.
        patches_info_json : json
            file containing info for patches, generated automatically when 
            patches are extracted.

        Returns
        -------
        numpy_array
            Fused image.

        '''
        with open(patches_info_json, 'r') as file:        
            patch_info = json.load(file)
        image_h, image_w, image_d = patch_info['image_res']
        patch_size = patch_info['size']
        fusion_image = np.zeros((image_h + patch_size[0], image_w + patch_size[1], image_d + patch_size[2]))
        fusion_matrix = np.zeros((image_h + patch_size[0], image_w + patch_size[1], image_d + patch_size[2]), dtype=np.uint8)
        
        for i in range(patch_info['len']):
            x, y, z = patch_info[str(i)]
            patch = patches[i]
            assert len(patch.shape) == 3, "The patch has more or less than 3 dimensions."
            fusion_image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += patch
            fusion_matrix[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]] += 1
            print(f"\rFusing patches...[{i/patch_info['len']:.2%}]", end='')
        print()

        fusion_matrix = np.where(fusion_matrix == 0, 1, fusion_matrix)
        # Averaging the patches values
        fusion_image = fusion_image / fusion_matrix
        # Saving fusion matrix used for averaging...
        nrrd.write(f'{self.result_folder}/fusion_matrix.nrrd', fusion_matrix)

        return fusion_image[:image_h, :image_w, :image_d]
        
    def save_result(self, results):
        """
        Transform the results to NRRD and save it in result folder.

        Parameters
        ----------
        results : tensor
            output of model.predict().

        """
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        print('Result shape: ', results.shape)
        for result, name in zip(results, self.test_list):
            result = np.reshape(result, result.shape[:3])
            nrrd.write(f'{self.result_folder}/{name}', result, header=None)
            print(result.shape, f'{self.result_folder}/{name} saved.')
            
    def save_result_patches(self, results):
        """
        Fuse patches to whole image, transform the image to NRRD and 
        save it in result folder.

        Parameters
        ----------
        results : tensor
            output of model.predict().

        """
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        results = np.squeeze(results, axis=-1)
        patches = []
        for patch in results:
            patches.append(patch)
        fused_image = self._fuse_patches(patches, f'{self.result_folder}/{self.test_list[0]}.json')
        print("Saving nrrd image...")
        nrrd.write(f'{self.result_folder}/{self.test_list[0]}', fused_image)
            
    def get_train_size(self):
        return self.train_size

    def get_val_size(self):
        return self.val_size

    def get_test_size(self):
        return self.test_size
    
    def get_input_size(self):
        return self.input_size
