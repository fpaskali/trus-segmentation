#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:44:53 2020

@author: paskali
 
A script to remove false positive results, that worsen Hausdorff and average surface distance.
Results are posprocessed and only the biggest region is keeped on each slide. All other smaller
regions are removed. This give better results and it tends to improve all scores.

"""

import nrrd, os, argparse
import numpy as np
from skimage.measure import label, regionprops


def _remove_small_false_positive_regions(image):
    image = np.where(image > 0.5, 1, 0)
    label_image = np.zeros(image.shape)
    
    for slide in range(image.shape[-1]):
        label_slide = label(image[:,:,slide])
        for region in regionprops(label_slide):
            if region.area > 500:
                label_val = region.label
                
                label_image[:,:,slide] = np.where(label_slide==label_val, 1, 0)

    return label_image

def postprocess_images(result_folder, output_folder):
    """
    Load images in result folder, analyse each slide, and keep only the biggest
    region.

    Parameters
    ----------
    result_folder : str
        path to result folder.
    output_folder : str
        path to output folder.

    Returns
    -------
    None.

    """
    image_names = []
    for img_name in os.listdir(result_folder):
        if img_name.endswith('.nrrd'):
            image_names.append(img_name)
    
    image_names.sort()        
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for img_name in image_names:
        image = nrrd.read(f'{result_folder}/{img_name}')[0]
        image = _remove_small_false_positive_regions(image)
        nrrd.write(f'{output_folder}/{img_name}', image)


def main():

    parser = argparse.ArgumentParser(description="Postprocessing result to enhance them further.")
    parser.add_argument('-result', help='path to the results folder.')
    parser.add_argument('-output', help='path to the output folder.')
    args = parser.parse_args()

    postprocess_images(args.result, args.output)

if __name__ == '__main__':
    main()
