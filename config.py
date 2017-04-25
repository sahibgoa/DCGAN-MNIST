"""
 Configuration file for the GAN
"""

import numpy as np

MAX_EPOCHS = 100  # maximal number of epochs to run
ACCURACY_FREQUENCY = 50  # how often to print accuracy
SAVE_FREQUENCY = 50
BATCH_SIZE = 1024  # how many instances per batch
IMAGE_SHAPE = [28, 28, 1]
ETA = 0.0002
STDDEV = 0.1
TRAIN_SIZE = 60000
DIM_IM = np.prod(IMAGE_SHAPE)
DIM_Z = 100
DIM_H1 = 150
DIM_H2 = 300
KEEP_PROB = 0.7