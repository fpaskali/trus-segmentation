#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:58:48 2019

@author: paskali
"""
import os, time
import matplotlib.pyplot as plt
import tensorflow as tf
from data import DataManager
from models.vnet import model1, model2, model3
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.backend import clear_session

# Setting dynamically grow memory for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Force Tensorflow to use CPU
def force_CPU():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

def train(epochs, image_size, model_name ="model1", batch_size=100, patch_size=None, stride_size=None,
          train_folder='data/train', val_folder='data/val', test_folder='data/test',
          result_folder='data/results/{model_name}', force_cpu=False):
    """
    Train a model in a predefined number of epochs. After training a graph with
    model metrics will be saved in data/metrics/{model_name}. The weights are
    saved in HDF5 format.

    Parameters
    ----------
    epochs : int
        Number of epochs to train the model.
    image_size : tuple
        The size of the whole image, or the image from which the patches will
        be extracted.
    model_name : str, optional
        Choose between 3 models from models.vnet. The default is "model1".
    batch_size : int, optional
        The number of images in mini-batch, that will be preloaded in memory. The default is 100.
    patch_size : tuple, optional
        the size of the patch. E.g. (128,128,32). The default is None.
    stride_size : tuple, optional
        the size of the stride. E.g. (64,64,16). The default is None.
    train_folder : str, optional
        path to train folder, with image and mask subfolders. The default is 'data/train'.
    val_folder : str, optional
        path to val folder, with image and mask subfolders. The default is 'data/val'.
    test_folder : str, optional
        path to test folder, with image and mask subfolders. The default is 'data/test'.
    result_folder : str, optional
        path to result folder. The default is 'data/results/{model_name}'.
    force_cpu : bool, optional
        A switch to enable CPU training. Not recommended, because GPU training
        is faster. The default is False.
    """

    if force_cpu:
        force_CPU()
        
    data_manager = DataManager(train_folder, val_folder, test_folder, result_folder, 
                               image_size, patch_size, stride_size)

    train_size = data_manager.get_train_size()
    val_size = data_manager.get_val_size()
    if (patch_size and stride_size):
        train_data = data_manager.train_patches_generator(epochs)
        val_data = data_manager.val_patches_generator(epochs)
    else:   
        train_data = data_manager.train_generator(epochs, batch_size)
        val_data = data_manager.val_generator(epochs=epochs)
    
    if model_name == 'model1':
        model = model1(input_size=data_manager.get_input_size())
    elif model_name == 'model2':
        model = model2(input_size=data_manager.get_input_size())
    elif model_name == 'model3':
        model = model3(input_size=data_manager.get_input_size())

    model_checkpoint = ModelCheckpoint(f'{model_name}_checkpoint.hdf5', monitor='loss', 
                                       verbose=1, save_best_only=True, 
                                       save_weights_only=False,
                                       mode='auto', save_freq=train_size)

    history = model.fit(x=train_data, validation_data=val_data,
                        steps_per_epoch=train_size, validation_steps=val_size,
                        epochs=epochs, verbose=1, callbacks=[model_checkpoint])
    
    model.save_weights(f'{model_name}_weights.hdf5')
    
    if not os.path.exists(f'metrics/{model_name}'):
        os.makedirs(f'metrics/{model_name}')
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Vnet Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(f'metrics/{model_name}/{time.strftime("%Y%m%d_%H%M")}_training_loss.png')
    plt.close()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Vnet Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(f'metrics/{model_name}/{time.strftime("%Y%m%d_%H%M")}_training_acc.png')
    plt.close()
    
    clear_session()
     
def test(image_size, model_name ="model1", batch_size=100, patch_size=None, stride_size=None,
          train_folder='data/train', val_folder='data/val', test_folder='data/test',
          result_folder='data/results/{model_name}', force_cpu=False):
    """
    Use weights stored in HDF5 format to perform segmentation. A NRRD results 
    will be saved in data/results/{model_name}

    Parameters
    ----------
    image_size : tuple
        The size of the whole image, or the image from which the patches will
        be extracted.
    model_name : str, optional
        Choose between 3 models from models.vnet. The default is "model1".
    batch_size : int, optional
        The number of images in mini-batch, that will be preloaded in memory. The default is 100.
    patch_size : tuple, optional
        the size of the patch. E.g. (128,128,32). The default is None.
    stride_size : tuple, optional
        the size of the stride. E.g. (64,64,16). The default is None.
    train_folder : str, optional
        path to train folder, with image and mask subfolders. The default is 'data/train'.
    val_folder : str, optional
        path to val folder, with image and mask subfolders. The default is 'data/val'.
    test_folder : str, optional
        path to test folder, with image and mask subfolders. The default is 'data/test'.
    result_folder : str, optional
        path to result folder. The default is 'data/results/{model_name}'.
    force_cpu : bool, optional
        A switch to enable CPU training. Not recommended, because GPU training
        is faster. The default is False.
    """
    if force_cpu:
        force_CPU()
        
    data_manager = DataManager(train_folder, val_folder, test_folder, result_folder, 
                               image_size, patch_size, stride_size)
    
    if model_name == 'model1':
        model = model1(input_size=data_manager.get_input_size(), weights=f'{model_name}_weights.hdf5')
    elif model_name == 'model2':
        model = model2(input_size=data_manager.get_input_size(), weights=f'{model_name}_weights.hdf5')
    elif model_name == 'model3':
        model = model3(input_size=data_manager.get_input_size(), weights=f'{model_name}_weights.hdf5')
    
    test_size = data_manager.get_test_size()
    if (patch_size and stride_size):
        test_data = data_manager.test_patches_generator()
        results = model.predict_generator(test_data, steps=test_size, verbose=1)
        data_manager.save_result(results)
    else:   
        test_data = data_manager.test_generator()
        results = model.predict_generator(test_data, steps=test_size, verbose=1)
        data_manager.save_result_patches(results)

    clear_session()


