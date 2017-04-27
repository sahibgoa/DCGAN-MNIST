'''
 Configuration file for the GAN.
'''

import numpy as np

SAVE_PATH = 'data/out/' # save path
IMAGE_SHAPE = [28, 28] # dimensions of image in MNIST dataset
SAMPLE_SIZE = 1 # size of sample to save every epoch

TRAIN_SIZE = 60000 # size of MNIST training set
MAX_EPOCHS = 500  # maximal number of epochs to run, 100 is probably enough
BATCH_SIZE = 128  # number of instances per batch
ETA = 0.0001 # learning rate
STDDEV = 0.1 # default standard deviation for weights, sensitive

DIM_IM = np.prod(IMAGE_SHAPE) # size of flattened image
DIM_Z = 100 # size of input to generator
DIM_H1 = 150 # size of 1st hidden layer in generator
DIM_H2 = 300 # size of 2nd hidden layer in generator

KEEP_PROB = 0.5 # for drop out
BETA1 = 0.5 # parameter for optimizer
LEAK = 0.2 # slope of leaky relu

RANDOM_MEAN = 0.0 # mean for random input to generator
RANDOM_STDDEV = 1. # standard deviation for random input to generator

USE_XAVIER = True # use xavier initializer 
USE_BIAS = False # use biases, might be better without
USE_L1 = False # Use L1 regularization
USE_L2 = False # Use L2 regularization
LAMBDA = 0.01 # regularization weight