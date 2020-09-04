#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:25:24 2019

@author: paskali
"""
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv3D, PReLU, add, Conv3DTranspose, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def model3(input_size = (128,128,128,1), weights = None, show_summary=False):
    inputs = Input(input_size)
    conv1_1 = Conv3D(filters=16, kernel_size=(5,5,5), activation=None, padding = 'same', strides =(1,1,1))(inputs)
    conv1_1 = PReLU()(BatchNormalization()(conv1_1))
    conv1_2 = Conv3D(filters=16, kernel_size=(5,5,5), activation=None, padding = 'same', strides =(1,1,1))(conv1_1)
    conv1_2 = PReLU()(BatchNormalization()(conv1_2))
    sum_first_layer = add([conv1_1, conv1_2])
    
    downconv1 = Conv3D(filters=32, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_first_layer)
    downconv1 = PReLU()(BatchNormalization()(downconv1))
    downconv1 = Dropout(0.25)(downconv1)
    
    conv2_1 = PReLU()(BatchNormalization()(Conv3D(filters=32, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(downconv1)))
    conv2_2 = PReLU()(BatchNormalization()(Conv3D(filters=32, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv2_1)))
    sum_second_layer = add([downconv1, conv2_2])
    
    downconv2 = Conv3D(filters=64, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_second_layer)
    downconv2 = PReLU()(BatchNormalization()(downconv2))
    downconv2 = Dropout(0.5)(downconv2)
    
    conv3_1 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(downconv2)))
    conv3_2 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv3_1)))
    conv3_3 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv3_2)))
    sum_third_layer = add([downconv2, conv3_3])
    
    downconv3 = Conv3D(filters=128, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_third_layer)
    downconv3 = PReLU()(BatchNormalization()(downconv3))
    downconv3 = Dropout(0.5)(downconv3)
    
    conv4_1 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(downconv3)))
    conv4_2 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv4_1)))
    conv4_3 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv4_2)))
    conv4_3 = Dropout(0.5)(conv4_3)
    sum_forth_layer = add([downconv3, conv4_3])
    
    # downconv4 = Conv3D(filters=256, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_forth_layer)
    # downconv4 = PReLU()(BatchNormalization()(downconv4))
    
    # conv5_1 = PReLU()(BatchNormalization()(Conv3D(filters=256, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(downconv4)))
    # conv5_2 = PReLU()(BatchNormalization()(Conv3D(filters=256, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv5_1)))
    # conv5_3 = PReLU()(BatchNormalization()(Conv3D(filters=256, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv5_2)))
    # sum_fifth_layer = add([downconv4, conv5_3])
    
    # upconv5 = PReLU()(BatchNormalization()(Conv3DTranspose(filters=128, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_fifth_layer)))
    
    # concat4 = Concatenate(axis=-1)([sum_forth_layer,upconv5])
    
    # conv4_4 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(concat4)))
    # conv4_5 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv4_4)))
    # conv4_6 = PReLU()(BatchNormalization()(Conv3D(filters=128, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv4_5)))
    # sum_forth_layer_up = add([conv4_6, upconv5])
    
    upconv4 = Conv3DTranspose(filters=64, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_forth_layer)
    upconv4 = PReLU()(BatchNormalization()(upconv4))
    
    concat3 = Concatenate(axis=-1)([sum_third_layer, upconv4])
    concat3 = Dropout(0.5)(concat3)
    
    conv3_4 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(concat3)))
    conv3_5 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv3_4)))
    conv3_6 = PReLU()(BatchNormalization()(Conv3D(filters=64, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv3_5)))
    sum_third_layer_up = add([conv3_6, upconv4])
    
    upconv3 = Conv3DTranspose(filters=32, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_third_layer_up)
    upconv3 = PReLU()(BatchNormalization()(upconv3))
    upconv3_1 = Conv3DTranspose(filters=16, kernel_size=(2,2,2), padding='valid', strides=(2,2,2))(upconv3)
    
    concat2 = Concatenate(axis=-1)([sum_second_layer, upconv3])
    concat2 = Dropout(0.5)(concat2)
    
    conv2_3 = PReLU()(BatchNormalization()(Conv3D(filters=32, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(concat2)))
    conv2_4 = PReLU()(BatchNormalization()(Conv3D(filters=32, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(conv2_3)))
    sum_second_layer_up = add([conv2_4, upconv3])
    
    upconv2 = Conv3DTranspose(filters=16, kernel_size=(2,2,2), activation=None, padding='valid', strides=(2,2,2))(sum_second_layer_up)
    upconv2 = PReLU()(BatchNormalization()(upconv2))
    
    concat1 = Concatenate(axis=-1)([sum_first_layer, upconv2])
    concat1 = Dropout(0.5)(concat1)
    
    conv1_3 = PReLU()(BatchNormalization()(Conv3D(filters=16, kernel_size=(5,5,5), activation=None, padding='same', strides=(1,1,1))(concat1)))
    sum_first_layer_up = add([conv1_3, upconv2])
            
    concat_prefinal = Concatenate(axis=-1)([upconv3_1, upconv2, sum_first_layer_up])
    
    final = PReLU()(BatchNormalization()(Conv3D(filters=16, kernel_size=(1,1,1), strides=(1,1,1))(concat_prefinal)))
    output = Conv3D(filters=1, kernel_size=(1,1,1), activation='sigmoid')(final)
    
    model = Model(inputs=inputs, outputs=output)
    
    optimizer = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    
    model.compile(optimizer=optimizer, loss = jaccard_distance_loss,  metrics = ['accuracy'])
    
    if show_summary:
        model.summary()
    
    if(weights):
        model.load_weights(weights)
    
    return model

def hybrid_loss(y_true, y_pred, bc_weight=0.5, dc_weight=1, dc_smooth=1):
    y_pred = tf.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    dice_coef =  (2. * intersection + dc_smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + dc_smooth)
    bc = K.binary_crossentropy(y_true, y_pred)
    dc = 1 - dice_coef
    
    return bc_weight * bc + dc_weight * dc

def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    y_pred = tf.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dc_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    y_pred = tf.constant(y_pred) if not tf.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth
