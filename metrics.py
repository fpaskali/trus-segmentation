#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:09:33 2019

@author: paskali
"""
import os, csv, argparse
import nrrd
import numpy as np
from medpy.metric.binary import dc, jc, hd, asd

def _create_metrics_csv(filename):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['image', 'dc', 'js', 'hd', 'asd']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()

def save_metrics_to_csv(filename, result_folder='data/results', reference_folder='data/test/mask',
                        voxelspacing=(0.538624, 0.538624, 0.5)):
    """
    Measure various metrics between images in result and reference folder. The
    images should have same name.

    Parameters
    ----------
    filename : str
        name used for saving csv file.
    result_folder : str, optional
        path to result folder. The default is 'data/results'.
    reference_folder : str, optional
        path to reference folder. The default is 'data/test/mask'.
    voxelspacing : tuple, optional
        voxel spacing in mm. The default is (0.538624, 0.538624, 0.5).

    Returns
    -------
    None.

    """
    _create_metrics_csv(filename)
    
    name_list = os.listdir(reference_folder)
    
    with open(filename, 'a') as csvfile:
        for name in name_list:
            res = nrrd.read(os.path.join(result_folder, name))[0]
            ref = nrrd.read(os.path.join(reference_folder, name))[0]
            res = np.where(res > 0.5, 1, 0)
            ref = np.where(ref > 0.5, 1, 0)

            metrics_dict = {}
            metrics_dict['image'] = name
            metrics_dict['dc'] = dc(res, ref)
            metrics_dict['js'] = jc(res, ref)
            metrics_dict['hd'] = hd(res, ref, voxelspacing=voxelspacing)
            metrics_dict['asd'] = asd(res, ref, voxelspacing=voxelspacing)
            
            writer = csv.DictWriter(csvfile, fieldnames=list(metrics_dict.keys()))
            writer.writerow(metrics_dict)

def main():
    parser = argparse.ArgumentParser(description="Measures DC, JC, Hausdorff, ASD and saves in csv")
    parser.add_argument('-result', help='path to results folder.', required=True)
    parser.add_argument('-ref', help='path to reference folder.', required=True)
    parser.add_argument('-vs', help='voxel spacing', type=tuple, default=(0.538624, 0.538624, 0.5))
    parser.add_argument('-csv', help='name of csv file.', required=True)

    args = parser.parse_args()

    save_metrics_to_csv(args.csv, args.result, args.ref, args.vs)

if __name__ == '__main__':
    main()
