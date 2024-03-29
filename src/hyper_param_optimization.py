#usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:58:48 2019

@author: paskali
"""
import os
import tensorflow as tf
from data import DataManager
from augment import augment_generator_probability
from models.hyper_vnet import VnetHyperModel
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.backend import clear_session
from kerastuner.tuners import BayesianOptimization

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
    
class MyTuner(BayesianOptimization):
    """
    A custom tuner based on BayesianOptimization tuner that introduce also the
    augmentation parameters as hyperparameters.
    """
    
    def run_trial(self, trial, x, *args, **kwargs):
        hp = trial.hyperparameters       
        train_ds = augment_generator_probability(x, 
                                                 factor=hp.Fixed('factor', 15),
                                                 rotate_p=hp.Float('rotate_prob', 0.1, 1., default=0.5),
                                                 deform_p=hp.Float('deform_prob', 0.1, 1., default=0.5), 
                                                 filters_p=hp.Float('filters_prob', 0.1, 1., default=0.5),
                                                 mean_filter_p=0.33, 
                                                 median_filter_p=0.33, 
                                                 gauss_filter_p=0.33,
                                                 epochs=150)
        
        super(MyTuner, self).run_trial(trial, train_ds, *args, **kwargs)

def tune(image_size, epochs, fact = 15, train_folder='data/train', 
          val_folder='data/val', force_cpu=False):
    """
    Tune the parameters and print the best 10 models.

    Parameters
    ----------
    image_size : tuple
        The size of the whole image, or the image from which the patches will
        be extracted.
    fact : int, optional
        The factor to increase the sample size.
    train_folder : str, optional
        path to train folder, with image and mask subfolders. The default is 'data/train'.
    val_folder : str, optional
        path to val folder, with image and mask subfolders. The default is 'data/val'.
    force_cpu : bool, optional
        A switch to enable CPU training. Not recommended, because GPU training
        is faster. The default is False.
    """

    if force_cpu:
        force_CPU()
        
    data_manager = DataManager(train_folder, val_folder, image_shape=image_size)
    
    train_size = data_manager.get_train_size() * fact
    val_size = data_manager.get_val_size()
    train_data = data_manager.train_generator()
    val_data = data_manager.val_generator()
    
    
    hypermodel = VnetHyperModel(input_shape=data_manager.get_input_size(), num_classes=1)


    model_checkpoint = EarlyStopping(monitor="val_loss",
                                     min_delta=0,
                                     patience=3,
                                     verbose=1,
                                     mode="auto",
                                     baseline=None,
                                     restore_best_weights=True)
    
    tuner = MyTuner(
            hypermodel=hypermodel,
            objective="val_accuracy",
            max_trials=50,
            directory="optimization",
            project_name="second_opti")
    
    tuner.search_space_summary()

    tuner.search(x=train_data, validation_data=val_data, validation_steps=val_size,
                epochs=epochs, verbose=1, steps_per_epoch=train_size, callbacks=[model_checkpoint])

    tuner.results_summary()
    
    clear_session()
             
if __name__ == '__main__':
    tune(image_size=(128,128,128), epochs=150)
