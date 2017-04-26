"""
 Configuration file for the GAN.
"""

import numpy as np

SAVE_PATH = 'data/out/sample_%04d.jpg'
MAX_EPOCHS = 1000  # maximal number of epochs to run
BATCH_SIZE = 1024  # how many instances per batch
ETA = 0.0001 # learning rate
STDDEV = 0.1 # default standard deviation
TRAIN_SIZE = 60000 # size of training set
IMAGE_SHAPE = [28, 28] # dimensions of image in MNIST dataset
DIM_IM = np.prod(IMAGE_SHAPE) # size of flattened image
DIM_Z = 100 # size of input to generator
DIM_H1 = 150 
DIM_H2 = 300
KEEP_PROB = 0.5 # for drop out