"""
 Configuration file for the GAN
"""

import numpy as np

DATA_PATH = 'data/mnist/'
MAX_EPOCHS = 100  # maximal number of epochs to run
ACCURACY_FREQUENCY = 50  # how often to print accuracy
SAVE_FREQUENCY = 50
BATCH_SIZE = 1024  # how many instances per batch
IMAGE_SHAPE = [28, 28, 1]

STRIDE = [1, 2, 2, 1]
ETA = 0.0002
STDDEV = 0.02
DATASET_SIZE = 50000

# GAN size params
DIM_Z = 100
DIM_Y = 10
DIM_W1 = 1024
DIM_W2 = 128
DIM_W3 = 64
DIM_IM = np.prod(IMAGE_SHAPE)

DIM_H1 = 150
DIM_H2 = 300

KEEP_PROB = 0.7