#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:35:20 2020

@author: paskali
"""
import random, time
import numpy as np
from preprocessing import ImageProcessor
from augment import augment_images
from train_test import train, test
from postprocessing import postprocess_images
from metrics import save_metrics_to_csv


def wholeimage_training(model_name, epochs, kfold=10):   
    times = []
    print('PREPROCESSING...')
    processor = ImageProcessor('data/raw_train')
    processor.crop_images((512,512,64))
    processor.resize_images((128,128,128))
    processor.normalize()
    image_names = processor.image_names
    round = 1
    for train_idx, val_idx in cv_split(image_names, kfold, shuffle=True):
        start = time.time()
        if not times:
            print(f'Round: {round}/{kfold} ETA?')
        else:
            print(f'Round: {round}/{kfold} ETA: {parse_time(int(np.mean(times)*(kfold-round+1)))}')
        processor.create_root_skeleton(preserve_results=True)
        processor.save_images('data/train', train_idx)
        processor.save_images('data/val', val_idx)
        augment_images('data/train')
        train(epochs, (128,128,128), model_name)
        test((128,128,128), model_name, test_folder='data/val')
        round += 1
        postprocess_images(f'data/results/{model_name}', f'data/postprocessed/{model_name}')
        save_metrics_to_csv(f'metrics/{model_name}/wi_training.csv',f'data/postprocessed/{model_name}', 'data/val/mask')
        end = time.time()
        times.append(end-start)
    save_training_info(f'metrics/{model_name}/training_report.txt', model_name, epochs, kfold, parse_time(int(np.sum(times))))

def train_test_training(model_name, epochs, val_size=0.2):   
    print('PREPROCESSING...')
    processor = ImageProcessor('data/raw_train')
    processor.crop_images((512,512,64))
    processor.resize_images((128,128,128))
    processor.normalize()
    image_names = processor.image_names
    round = 1
    train_idx, val_idx = train_val_split(image_names,val_size,random_seed='Filip')
    start = time.time()
    processor.create_root_skeleton(preserve_results=True)
    processor.save_images('data/train', train_idx)
    processor.save_images('data/val', val_idx)
    augment_images('data/train')
    train(epochs, (128,128,128), model_name)
    test((128,128,128), model_name, test_folder='data/val')
    round += 1
    postprocess_images(f'data/results/{model_name}', f'data/postprocessed/{model_name}')
    save_metrics_to_csv(f'metrics/{model_name}/tt_training.csv',f'data/postprocessed/{model_name}', 'data/val/mask')
    end = time.time()
    save_training_info(f'metrics/{model_name}/training_report.txt', model_name, epochs, kfold, parse_time(end-start))

def patch_wise_training(model_name, epochs, kfold=10):
    print('PREPROCESSING...')
    processor = ImageProcessor('data/raw_train')
    processor.crop_images((512,512,64))
    processor.normalize()
    image_names = processor.image_names
    for train_idx, val_idx in cv_split(image_names, kfold, shuffle=True):
        processor.create_root_skeleton(preserve_results=True)
        processor.save_images('data/train', train_idx)
        processor.save_images('data/val', val_idx)
        augment_images('data/train')
        train(epochs, (512,512,64), model_name, patch_size=(128,128,32), stride_size=(64,64,16))
        test((512,512,64), model_name, patch_size=(128,128,32), stride_size=(64,64,16), test_folder='data/val')
    postprocess_images('data/results', 'data/results/postprocessed')
    save_metrics_to_csv(f'metrics/pw_training.csv','data/results/postprocessed', 'data/val/mask')

def whole_image_testing(model_name):
    print('PREPROCESSING...')
    processor = ImageProcessor('data/raw_test', test_set=True)
    processor.crop_images((512,512,64))
    processor.resize_images((128,128,128))
    processor.normalize(norm_params='norm_params.json')
    processor.save_images('data/test')
    test((128,128,128), model_name, test_folder='data/test')
    postprocess_images('data/results', 'data/results/postprocessed')
    save_metrics_to_csv(f'metrics/wi_training.csv','data/results/postprocessed', 'data/test/mask')

def patch_wise_testing(model_name):
    print('PREPROCESSING...')
    processor = ImageProcessor('data/raw_test', test_set=True)
    processor.crop_images((512,512,64))
    processor.normalize(norm_params='norm_params.json')
    processor.save_images('data/test')
    test((512,512,64), model_name, patch_size=(128,128,32), stride_size=(64,64,16), test_folder='data/test')
    postprocess_images('data/results', 'data/results/postprocessed')
    save_metrics_to_csv(f'metrics/pw_training.csv','data/results/postprocessed', 'data/test/mask')

# Cross-validation function    
def cv_split(data, k_fold, shuffle=False):
    """
    Split the data in K number of folds for cross validation. Return indices of 
    train and test set.

    Parameters
    ----------
    data : list
        it could be list of any items.
    k_fold : int
        the number of folds.
    shuffle : bool, optional
        If true, shuffles the order of the items in the list. The default is False.

    Returns
    -------
    splits : list
        List of tuples of train and test set indices.

    """
    data_idx = [*range(len(data))]
    if shuffle: random.shuffle(data_idx)
    assert k_fold > 1, "ERROR: K_fold cannot be 1 or less."
    assert k_fold <= len(data_idx), "ERROR: K_fold is greater than sample size."
    kfold_size = len(data_idx) // k_fold 
    folds = [kfold_size for x in range(k_fold)]
    
    pos = 0
    while np.sum(folds) < len(data_idx):
        folds[pos] += 1
        pos += 1
    assert np.sum(folds) == len(data_idx), "Splitting error"
          
    splits = []
    pos = 0
    for fold_size in folds:
        splits.append((data_idx[0:pos]+data_idx[pos+fold_size:], data_idx[pos:pos+fold_size]))
        pos += fold_size
        
    return splits

def train_val_split(data, val_size, random_seed=None):
    data_idx = [*range(len(data))]
    random.seed(random_seed)
    random.shuffle(data_idx)
    
    if type(val_size) == float:
        val_size = round(len(data_idx) * val_size)
    elif type(val_size) == int:
        val_size = val_size
    else:
        raise ValueError('val_size should be int or float')
        
    val = data_idx[:val_size]
    train = data_idx[val_size:]
    
    return train, val

def parse_time(seconds):
    minutes = seconds / 60
    rest_seconds = int(seconds % 60)
    
    hours = int(minutes / 60)
    rest_minutes = int(minutes % 60)
    
    return f'{hours}:{rest_minutes:02}:{rest_seconds:02}'

def save_training_info(filename, model_name, epochs, kfolds, total_time):
    with open(filename, "w") as report:
        report.write(f"Model name: {model_name}\n")
        report.write(f"Number of epochs: {epochs}\n")
        report.write(f"Cross-validation k-folds: {kfolds}\n")
        report.write(f"Total time of training: {total_time}\n")

#%% Example usage
# Whole image training using Model 1, and epochs = 35. Training set should be located in 
# 'data/raw_train/image' and 'data/raw_train/mask'       
#wholeimage_training('model3', 35)

# Testing should be done after training the model. It uses weights generated 
# by train function. Testing set should be located in 'data/raw_test/image'
#whole_image_testing('model1')


# Patchwise training using Model 2, and epochs = 35. Training set should be located in 
# 'data/raw_train/image' and 'data/raw_train/mask'
#patch_wise_training('model2', 35)

# Testing should be done after training the model. It uses weights generated 
# by train function. Testing set should be located in 'data/raw_test/image'
#patch_wise_testing('model2')
# 
# wholeimage_training('model3', 50)
train_test_training('model3', 100)