# trus-segmentation

This is final fully commented version of the code. [Not tested]

It combines whole image and patch-wise training. Patch-wise training is executed when patch_size and stride_size are defined in train and test function, otherwise it will perform whole image training.

The functions for training and testing, as well as examples can be found in main.py.

It is important before training, unprocessed NRRD images and binary masks to be moved in 'data/raw_train/image', 'data/raw_train/mask' or for testing 'data/raw_test/image', 'data/raw_test/mask'. Images and binary mask should have same name.
