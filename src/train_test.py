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
from augment import augment_generator_probability
from models.dynamic_vnet import VnetDynamicModel
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
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

def train(model_name, image_size, epochs, hyper_par, train_folder='data/train', 
          val_folder='data/val', force_cpu=False):
    """
    Train a model in a predefined number of epochs. After training a graph with
    model metrics will be saved in data/metrics/{model_name}. The weights are
    saved in HDF5 format.

    Parameters
    ----------
    model_name : str, optional
        The name of the model. It is used for saving metrics and weights.
    image_size : tuple
        The shape of the numpy array.
    epochs : int
        Number of epochs to train the model.
    hyper_par : dict
        It contains all model hyperparameters.
    train_folder : str, optional
        path to train folder, with image and mask subfolders. The default is 'data/train'.
    force_cpu : bool, optional
        A switch to enable CPU training. Not recommended, because GPU training
        is faster. The default is False.
    """

    if force_cpu:
        force_CPU()
        
    data_manager = DataManager(train_folder, val_folder, image_shape=image_size)
    
    train_size = data_manager.get_train_size() * hyper_par["factor"]
    val_size = data_manager.get_val_size()
    train_data = data_manager.train_generator()
    val_data = data_manager.val_generator()
    
    model = VnetDynamicModel(input_shape=data_manager.get_input_size(), num_classes=1)

    model = model.build(l_rate=hyper_par["l_rate"], beta1=hyper_par["beta1"],beta2=hyper_par["beta2"],
                        ams_grad=hyper_par["ams_grad"], model_loss=hyper_par["loss"], fifth_level=hyper_par["fifth"])

    model_checkpoint = EarlyStopping(  monitor='val_loss', 
                                       min_delta=0,
                                       patience=3,
                                       verbose=1,
                                       mode="auto",
                                       baseline=None,
                                       restore_best_weights=True
                                     )
    
    augmented_train_data = augment_generator_probability(train_data, 
                                                 factor=hyper_par["factor"],
                                                 rotate_p=hyper_par["rotate"],
                                                 deform_p=hyper_par["deform"], 
                                                 filters_p=hyper_par["filters"],
                                                 mean_filter_p=0.33, 
                                                 median_filter_p=0.33, 
                                                 gauss_filter_p=0.33,
                                                 epochs=epochs)

    history = model.fit(x=augmented_train_data, validation_data=val_data,
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
    
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Vnet Model DSC')
    plt.ylabel('DSC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Valid'], loc='upper left')
    plt.savefig(f'metrics/{model_name}/{time.strftime("%Y%m%d_%H%M")}_training_dsc.png')
    plt.close()
    
    clear_session()
     
def test(model_name, image_size, hyper_par, test_folder='data/test',
          result_folder='data/results', force_cpu=False):
    """
    Use weights stored in HDF5 format to perform segmentation. A NRRD results 
    will be saved in data/results/{model_name}

    Parameters
    ----------
    model_name : str, optional
        The name of the model. It is used for saving metrics and weights.
    image_size : tuple
        The shape of the numpy array.
    hyper_par : dict
        It contains all model hyperparameters.
    test_folder : str, optional
        path to test folder, with image and mask subfolders. The default is 'data/test'.
    result_folder : str, optional
        path to result folder. The default is 'data/results/'.
    force_cpu : bool, optional
        A switch to enable CPU training. Not recommended, because GPU training
        is faster. The default is False.
    """
    if force_cpu:
        force_CPU()
        
    data_manager = DataManager(test_folder=test_folder, 
                               result_folder=f'{result_folder}/{model_name}', 
                               image_shape=image_size)
    
    model = VnetDynamicModel(input_shape=data_manager.get_input_size(), 
                             num_classes=1, weights=f"{model_name}_weights.hdf5")
    
    model = model.build(l_rate=hyper_par["l_rate"], beta1=hyper_par["beta1"],beta2=hyper_par["beta2"],
                        ams_grad=hyper_par["ams_grad"], model_loss=hyper_par["loss"], fifth_level=hyper_par["fifth"])
    
    test_size = data_manager.get_test_size()
    
    test_data = data_manager.test_generator()
    results = model.predict(test_data, steps=test_size, verbose=1)
    data_manager.save_result(results)

    clear_session()

# train("the_best_model_3.11.2020", (128,128,128), 150, hyper_par = {"factor":15,
#                                                                   "l_rate":0.0001,
#                                                                   "beta1":0.43649430628078034,
#                                                                   "beta2":0.5898459767675351,
#                                                                   "ams_grad":False,
#                                                                   "loss":"jaccard",
#                                                                   "fifth":True,
#                                                                   "rotate":0.533,
#                                                                   "deform":0.901,
#                                                                   "filters":0.370})

# train("second_best_model_3.11.2020", (128,128,128), 150, hyper_par={"factor":15,
#                                                                     "l_rate":0.001,
#                                                                     "beta1":0.14369651566686886,
#                                                                     "beta2":0.8290607750524758,
#                                                                     "ams_grad":False,
#                                                                     "loss":"jaccard",
#                                                                     "fifth":True,
#                                                                     "rotate":0.868,
#                                                                     "deform":0.130,
#                                                                     "filters":0.807})

# train("third_best_model_3.11.2020", (128,128,128), 150, hyper_par={"factor":15,
#                                                                     "l_rate":0.0001,
#                                                                     "beta1":0.5209463997379207,
#                                                                     "beta2":0.7764027455241465,
#                                                                     "ams_grad":False,
#                                                                     "loss":"jaccard",
#                                                                     "fifth":False,
#                                                                     "rotate":0.268,
#                                                                     "deform":0.958,
#                                                                     "filters":0.688})

test("the_best_model_12.11.2020", (128,128,128),  test_folder="data/val", hyper_par = {"factor":15,
                                                                                    "l_rate":0.0001,
                                                                                    "beta1":0.43649430628078034,
                                                                                    "beta2":0.5898459767675351,
                                                                                    "ams_grad":False,
                                                                                    "loss":"jaccard",
                                                                                    "fifth":True,
                                                                                    "rotate":0.533,
                                                                                    "deform":0.901,
                                                                                    "filters":0.370})

# test("second_best_model_3.11.2020", (128,128,128), test_folder="data/val", hyper_par={"factor":15,
#                                                                                     "l_rate":0.001,
#                                                                                     "beta1":0.14369651566686886,
#                                                                                     "beta2":0.8290607750524758,
#                                                                                     "ams_grad":False,
#                                                                                     "loss":"jaccard",
#                                                                                     "fifth":True,
#                                                                                     "rotate":0.868,
#                                                                                     "deform":0.130,
#                                                                                     "filters":0.807})

# test("third_best_model_3.11.2020", (128,128,128), test_folder="data/val", hyper_par={"factor":15,
#                                                                                     "l_rate":0.0001,
#                                                                                     "beta1":0.5209463997379207,
#                                                                                     "beta2":0.7764027455241465,
#                                                                                     "ams_grad":False,
#                                                                                     "loss":"jaccard",
#                                                                                     "fifth":False,
#                                                                                     "rotate":0.268,
#                                                                                     "deform":0.958,
#                                                                                     "filters":0.688})

# test("the_best_model_3.11.2020", (128,128,128), test_folder="data/test", hyper_par = {"factor":15,
#                                                                                     "l_rate":0.0001,
#                                                                                     "beta1":0.43649430628078034,
#                                                                                     "beta2":0.5898459767675351,
#                                                                                     "ams_grad":False,
#                                                                                     "loss":"jaccard",
#                                                                                     "fifth":True,
#                                                                                     "rotate":0.533,
#                                                                                     "deform":0.901,
#                                                                                     "filters":0.370})